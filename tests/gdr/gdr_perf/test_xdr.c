/* some special machine types and scenarios XDR loopback: host DRAM <-> XPU HBM via ibverbs.
 *
 * No Mooncake. No NVIDIA perftest. some special machine types and scenarios's own path:
 *   xpu_malloc                              (libxpurt)
 *   cudaPointerGetAttributes -> hostPointer (XMLIR libcudart, same as peermem_map.cpp)
 *   ibv_reg_mr + RDMA WRITE/READ            (libibverbs)
 *
 *   WRITE: host DRAM  --RDMA WRITE-->  XPU HBM
 *   READ : host DRAM  <--RDMA READ---  XPU HBM
 *
 * Host QP is the initiator. XPU memory is the remote MR.
 *
 *   cc -O2 -Wall test_xdr.c -o test_xdr -libverbs -ldl
 *   ./test_xdr -d mlx5_bond_0 -g 0 -x 3 write
 *   ./test_xdr -d mlx5_bond_0 -x 3 -H write          # host DRAM RoCE, no XPU
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <getopt.h>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define HBM_BEGIN 0x4000000000ULL
#define HBM_END 0x5800000000ULL
#define DEFAULT_SIZE (1u << 20)
#define DEFAULT_ITERS 1000
#define DEFAULT_QPS 1
#define DEFAULT_TX_DEPTH 128
#define MAX_QPS 64

#define DIE(fmt, ...)                                                          \
    do {                                                                       \
        fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__);                    \
        exit(1);                                                               \
    } while (0)

typedef struct {
    int type;
    int device;
    void *devicePointer;
    void *hostPointer;
} CudaPointerAttributes;

struct xpu_api {
    void *h;
    int (*set_device)(int);
    int (*malloc3)(void **, uint64_t, int);
    int (*malloc2)(void **, uint64_t);
    int (*xfree)(void *);
    int (*xmemset)(void *, int, uint64_t);
};

struct cuda_api {
    void *h;
    int (*ptr_attr)(CudaPointerAttributes *, const void *);
};

struct rdma {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp **init_qp;
    struct ibv_qp **tgt_qp;
    int nqps;
    int tx_depth;
    struct ibv_mr *host_mr;
    struct ibv_mr *xpu_mr;
    union ibv_gid gid;
    int port;
    int gid_index;
    uint8_t link_layer;
    uint16_t lid;
    enum ibv_mtu mtu;
};

static void *xdlopen(const char *name) {
    void *h = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (h)
        fprintf(stderr, "loaded %s\n", name);
    return h;
}

static void load_xpu(struct xpu_api *a) {
    memset(a, 0, sizeof(*a));
    const char *cands[] = {"libxpurt.so.2", "libxpurt.so", "libxpurt.so.1", NULL};
    for (int i = 0; cands[i]; i++) {
        a->h = xdlopen(cands[i]);
        if (a->h)
            break;
    }
    if (!a->h)
        DIE("dlopen libxpurt failed: %s\n"
            "Need XMLIR XRE on LD_LIBRARY_PATH (torch_xmlir/xre/so).",
            dlerror());
    a->set_device = dlsym(a->h, "xpu_set_device");
    a->malloc3 = dlsym(a->h, "xpu_malloc");
    a->malloc2 = dlsym(a->h, "xpu_malloc");
    a->xfree = dlsym(a->h, "xpu_free");
    a->xmemset = dlsym(a->h, "xpu_memset");
    if (!a->set_device || !a->malloc3)
        DIE("libxpurt missing xpu_set_device / xpu_malloc");
}

static void load_xmlir_cudart(struct cuda_api *a) {
    memset(a, 0, sizeof(*a));
    const char *env = getenv("XMLIR_CUDART");
    const char *cands[8];
    int n = 0;
    if (env && env[0])
        cands[n++] = env;
    cands[n++] = "libcudart.so";
    cands[n++] = "libcudart.so.12";
    cands[n++] = "libcudart.so.11.0";
    cands[n] = NULL;
    for (int i = 0; cands[i]; i++) {
        a->h = xdlopen(cands[i]);
        if (a->h)
            break;
    }
    if (!a->h)
        DIE("dlopen XMLIR libcudart failed: %s\n"
            "Need torch_xmlir/xre/so on LD_LIBRARY_PATH, not NVIDIA CUDA.",
            dlerror());
    a->ptr_attr = dlsym(a->h, "cudaPointerGetAttributes");
    if (!a->ptr_attr)
        DIE("XMLIR libcudart has no cudaPointerGetAttributes (XDR map)");
}

static void *xpu_alloc(struct xpu_api *a, size_t nbytes) {
    void *p = NULL;
    int rc = -1;
    /* KLX kernels: xpu_malloc(&p, size, XPU_MEM_HBM=0) */
    rc = a->malloc3(&p, (uint64_t)nbytes, 0);
    if (rc != 0 || !p) {
        p = NULL;
        rc = a->malloc2(&p, (uint64_t)nbytes);
    }
    if (rc != 0 || !p)
        DIE("xpu_malloc(%zu) failed rc=%d", nbytes, rc);
    return p;
}

static void *xdr_host_pointer(struct cuda_api *c, void *dev) {
    CudaPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    int rc = c->ptr_attr(&attr, dev);
    if (rc != 0 || !attr.hostPointer)
        DIE("cudaPointerGetAttributes failed rc=%d hostPointer=%p "
            "(CUDA_ENABLE_P2P_NO_UVA=1 required for XDR)",
            rc, attr.hostPointer);
    fprintf(stderr,
            "XDR map  device=%p  hostPointer=%p  type=%d  attr.device=%d\n",
            attr.devicePointer ? attr.devicePointer : dev, attr.hostPointer,
            attr.type, attr.device);
    return attr.hostPointer;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_nsec * 1e-9 + (double)ts.tv_sec;
}

static void fill_pattern(void *p, size_t n, unsigned seed) {
    unsigned char *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (unsigned char)((i + seed) & 0xff);
}

static size_t mismatch(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    size_t m = 0;
    for (size_t i = 0; i < n; i++)
        if (x[i] != y[i])
            m++;
    return m;
}

static struct ibv_device *find_dev(const char *name) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list)
        DIE("ibv_get_device_list failed");
    struct ibv_device *found = NULL;
    for (int i = 0; i < n; i++) {
        if (!name || strcmp(ibv_get_device_name(list[i]), name) == 0) {
            found = list[i];
            break;
        }
    }
    if (!found) {
        fprintf(stderr, "Available RDMA devices:\n");
        for (int i = 0; i < n; i++)
            fprintf(stderr, "  %s\n", ibv_get_device_name(list[i]));
        DIE("RDMA device not found: %s", name ? name : "(null)");
    }
    return found;
}

static struct ibv_qp *make_qp(struct rdma *r) {
    struct ibv_qp_init_attr ia;
    memset(&ia, 0, sizeof(ia));
    ia.send_cq = r->cq;
    ia.recv_cq = r->cq;
    ia.cap.max_send_wr = (uint32_t)r->tx_depth + 16;
    ia.cap.max_recv_wr = 16;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ia.qp_type = IBV_QPT_RC;
    struct ibv_qp *qp = ibv_create_qp(r->pd, &ia);
    if (!qp)
        DIE("ibv_create_qp: %s", strerror(errno));
    return qp;
}

static void qp_to_init(struct ibv_qp *qp, int port) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = (uint8_t)port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &a,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS))
        DIE("QP -> INIT: %s", strerror(errno));
}

static void qp_to_rtr(struct rdma *r, struct ibv_qp *qp, uint32_t dest_qpn,
                      uint32_t psn) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = r->mtu;
    a.dest_qp_num = dest_qpn;
    a.rq_psn = psn;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer = 12;
    a.ah_attr.port_num = (uint8_t)r->port;
    a.ah_attr.sl = 0;
    if (r->link_layer == IBV_LINK_LAYER_ETHERNET) {
        a.ah_attr.is_global = 1;
        a.ah_attr.grh.hop_limit = 64;
        a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
        a.ah_attr.grh.dgid = r->gid;
    } else {
        a.ah_attr.dlid = r->lid;
        a.ah_attr.is_global = 0;
    }
    if (ibv_modify_qp(qp, &a,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        DIE("QP -> RTR: %s", strerror(errno));
}

static void qp_to_rts(struct ibv_qp *qp, uint32_t psn) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = psn;
    a.max_rd_atomic = 16;
    if (ibv_modify_qp(qp, &a,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC))
        DIE("QP -> RTS: %s", strerror(errno));
}

static void rdma_open(struct rdma *r, const char *devname, int port,
                      int gid_index, int nqps, int tx_depth) {
    memset(r, 0, sizeof(*r));
    r->port = port;
    r->gid_index = gid_index;
    r->nqps = nqps;
    r->tx_depth = tx_depth;
    struct ibv_device *dev = find_dev(devname);
    r->ctx = ibv_open_device(dev);
    if (!r->ctx)
        DIE("ibv_open_device(%s) failed", ibv_get_device_name(dev));
    fprintf(stderr, "opened RDMA device %s\n", ibv_get_device_name(dev));

    struct ibv_port_attr pa;
    if (ibv_query_port(r->ctx, (uint8_t)port, &pa))
        DIE("ibv_query_port: %s", strerror(errno));
    r->link_layer = pa.link_layer;
    r->lid = pa.lid;
    r->mtu = pa.active_mtu;

    if (ibv_query_gid(r->ctx, (uint8_t)port, gid_index, &r->gid))
        DIE("ibv_query_gid(port=%d,index=%d) failed", port, gid_index);

    r->pd = ibv_alloc_pd(r->ctx);
    if (!r->pd)
        DIE("ibv_alloc_pd: %s", strerror(errno));
    int cq_size = nqps * tx_depth * 2;
    if (cq_size < 128)
        cq_size = 128;
    r->cq = ibv_create_cq(r->ctx, cq_size, NULL, NULL, 0);
    if (!r->cq)
        DIE("ibv_create_cq: %s", strerror(errno));

    r->init_qp = calloc((size_t)nqps, sizeof(*r->init_qp));
    r->tgt_qp = calloc((size_t)nqps, sizeof(*r->tgt_qp));
    if (!r->init_qp || !r->tgt_qp)
        DIE("oom qp arrays");
    for (int i = 0; i < nqps; i++) {
        r->init_qp[i] = make_qp(r);
        r->tgt_qp[i] = make_qp(r);
        qp_to_init(r->init_qp[i], port);
        qp_to_init(r->tgt_qp[i], port);
        uint32_t psn = 0x1234 + (uint32_t)i;
        qp_to_rtr(r, r->init_qp[i], r->tgt_qp[i]->qp_num, psn);
        qp_to_rtr(r, r->tgt_qp[i], r->init_qp[i]->qp_num, psn);
        qp_to_rts(r->init_qp[i], psn);
        qp_to_rts(r->tgt_qp[i], psn);
    }
    fprintf(stderr, "QPs connected  nqps=%d  tx_depth=%d  gid_index=%d\n", nqps,
            tx_depth, gid_index);
}

static struct ibv_mr *reg_mr(struct ibv_pd *pd, void *addr, size_t len,
                             const char *what) {
    int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
              IBV_ACCESS_REMOTE_READ;
    struct ibv_mr *mr = ibv_reg_mr(pd, addr, len, acc);
    if (!mr)
        DIE("ibv_reg_mr(%s ptr=%p len=%zu): %s", what, addr, len,
            strerror(errno));
    fprintf(stderr, "registered %s  addr=%p  len=%zu  lkey=0x%x rkey=0x%x\n",
            what, addr, len, mr->lkey, mr->rkey);
    return mr;
}

static void post_rdma(struct ibv_qp *qp, struct ibv_mr *local, void *laddr,
                      struct ibv_mr *remote, void *raddr, size_t len,
                      enum ibv_wr_opcode op, uint64_t wr_id, int signaled) {
    struct ibv_sge sge = {
        .addr = (uintptr_t)laddr,
        .length = (uint32_t)len,
        .lkey = local->lkey,
    };
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = op;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = (uintptr_t)raddr;
    wr.wr.rdma.rkey = remote->rkey;
    if (ibv_post_send(qp, &wr, &bad))
        DIE("ibv_post_send: %s", strerror(errno));
}

static int poll_update(struct ibv_cq *cq, int *ccnt, int nqps, int max,
                       const char *what) {
    struct ibv_wc wc[64];
    int nwant = max < 64 ? max : 64;
    if (nwant < 1)
        return 0;
    int n = ibv_poll_cq(cq, nwant, wc);
    if (n < 0)
        DIE("ibv_poll_cq");
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS)
            DIE("%s completion status=%s(%d) wr_id=%" PRIu64, what,
                ibv_wc_status_str(wc[i].status), wc[i].status,
                (uint64_t)wc[i].wr_id);
        int q = (int)wc[i].wr_id;
        if (q < 0 || q >= nqps)
            DIE("%s bad wr_id=%d", what, q);
        ccnt[q]++;
    }
    return n;
}

static void poll_exact(struct ibv_cq *cq, int *ccnt, int nqps, int need,
                       const char *what) {
    int got = 0;
    for (int i = 0; i < 10000000 && got < need; i++) {
        int n = poll_update(cq, ccnt, nqps, need - got, what);
        got += n;
    }
    if (got < need)
        DIE("%s timed out waiting for %d completions (got %d)", what, need, got);
}

static void run_op(struct rdma *r, int is_write, void *host, void *remote,
                   size_t nbytes, int iters, unsigned seed, int seed_remote,
                   const char *remote_name) {
    unsigned char *expect = malloc(nbytes);
    if (!expect)
        DIE("oom");
    fill_pattern(expect, nbytes, seed);

    if (is_write) {
        memcpy(host, expect, nbytes);
        if (seed_remote)
            memset(remote, 0, nbytes);
    } else {
        if (seed_remote)
            memcpy(remote, expect, nbytes);
        memset(host, 0, nbytes);
    }

    enum ibv_wr_opcode op = is_write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    int dummy[MAX_QPS];
    memset(dummy, 0, sizeof(dummy));
    post_rdma(r->init_qp[0], r->host_mr, host, r->xpu_mr, remote, nbytes, op, 0,
              1);
    poll_exact(r->cq, dummy, r->nqps, 1, is_write ? "RDMA WRITE" : "RDMA READ");

    const unsigned char *got =
        is_write ? (unsigned char *)remote : (unsigned char *)host;
    size_t bad = mismatch(got, expect, nbytes);
    if (bad)
        DIE("%s correctness FAILED: %zu / %zu bytes differ",
            is_write ? "WRITE" : "READ", bad, nbytes);
    printf("%s correctness OK  %zu bytes  host DRAM %s %s\n",
           is_write ? "WRITE" : "READ", nbytes, is_write ? "->" : "<-",
           remote_name);

    /* perftest-style: keep tx_depth outstanding WRs on each QP. */
    int *scnt = calloc((size_t)r->nqps, sizeof(int));
    int *ccnt = calloc((size_t)r->nqps, sizeof(int));
    if (!scnt || !ccnt)
        DIE("oom");
    int tot_scnt = 0, tot_ccnt = 0;
    int depth = r->tx_depth;
    if (depth < 1)
        depth = 1;
    if (depth > iters)
        depth = iters;

    double t0 = now_s();
    while (tot_ccnt < iters) {
        for (int q = 0; q < r->nqps && tot_scnt < iters; q++) {
            while (scnt[q] - ccnt[q] < depth && tot_scnt < iters) {
                post_rdma(r->init_qp[q], r->host_mr, host, r->xpu_mr, remote,
                          nbytes, op, (uint64_t)q, 1);
                scnt[q]++;
                tot_scnt++;
            }
        }
        tot_ccnt += poll_update(r->cq, ccnt, r->nqps, iters - tot_ccnt, "bw");
    }
    double dt = now_s() - t0;
    double gbps = dt > 0 ? (8.0 * (double)nbytes * (double)iters / dt / 1e9) : 0;
    double mpps = dt > 0 ? ((double)iters / dt / 1e6) : 0;
    printf("#bytes  #iters  #QPs  tx_depth  BW average[Gbps]  msgRate[Mpps]\n");
    printf("%-8zu %-7d %-5d %-9d %-17.3f %-14.4f\n", nbytes, iters, r->nqps,
           depth, gbps, mpps);

    free(scnt);
    free(ccnt);
    free(expect);
}

__attribute__((visibility("default"))) int xdr_gdr_run(
    void *dev_ptr, size_t nbytes, const char *ibdev, int port, int gid_index,
    int nqps, int tx_depth, int iters, int do_write, int do_read, int seed_xpu) {
    if (!dev_ptr || nbytes == 0)
        DIE("xdr_gdr_run: null device pointer or zero size");
    if (nqps < 1 || nqps > MAX_QPS)
        DIE("xdr_gdr_run: nqps out of range");
    if (tx_depth < 1 || iters < 1)
        DIE("xdr_gdr_run: tx_depth/iters must be >= 1");
    if (!ibdev || !ibdev[0])
        ibdev = "mlx5_bond_0";

    uintptr_t daddr = (uintptr_t)dev_ptr;
    printf("torch/XPU data_ptr = %p  in_HBM=%s\n", dev_ptr,
           (daddr >= HBM_BEGIN && daddr < HBM_END) ? "yes" : "NO");

    struct cuda_api cuda;
    load_xmlir_cudart(&cuda);
    void *xpu_map = xdr_host_pointer(&cuda, dev_ptr);

    void *host = NULL;
    if (posix_memalign(&host, 4096, nbytes))
        DIE("posix_memalign host: %s", strerror(errno));
    memset(host, 0, nbytes);

    struct rdma rd;
    rdma_open(&rd, ibdev, port, gid_index, nqps, tx_depth);
    rd.host_mr = reg_mr(rd.pd, host, nbytes, "host-dram");
    rd.xpu_mr = reg_mr(rd.pd, xpu_map, nbytes, "torch-xdr-hostPointer");

    if (do_write)
        run_op(&rd, 1, host, xpu_map, nbytes, iters, 0xA5, seed_xpu, "XPU HBM");
    if (do_read)
        run_op(&rd, 0, host, xpu_map, nbytes, iters, 0x5A, seed_xpu, "XPU HBM");

    ibv_dereg_mr(rd.host_mr);
    ibv_dereg_mr(rd.xpu_mr);
    for (int i = 0; i < rd.nqps; i++) {
        ibv_destroy_qp(rd.init_qp[i]);
        ibv_destroy_qp(rd.tgt_qp[i]);
    }
    free(rd.init_qp);
    free(rd.tgt_qp);
    ibv_destroy_cq(rd.cq);
    ibv_dealloc_pd(rd.pd);
    ibv_close_device(rd.ctx);
    free(host);
    return 0;
}

#ifndef XDR_AS_LIB
static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [-d ibdev] [-g gpu] [-x gid] [-s size] [-n iters]\n"
            "          [-q qps] [-D tx_depth] [-i port] [-H]\n"
            "          [write|read|both]\n"
            "\n"
            "  default : XDR  host DRAM <-> XPU HBM\n"
            "  -H      : RoCE host DRAM <-> host DRAM (no XPU)\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *dev = getenv("RDMA_DEV");
    if (!dev || !dev[0])
        dev = "mlx5_bond_0";
    int gpu = getenv("GPU_ID") ? atoi(getenv("GPU_ID")) : 0;
    int gid_index = getenv("GID_INDEX") ? atoi(getenv("GID_INDEX")) : 3;
    int port = 1;
    size_t nbytes = DEFAULT_SIZE;
    int iters = DEFAULT_ITERS;
    int nqps = DEFAULT_QPS;
    int tx_depth = DEFAULT_TX_DEPTH;
    int host_only = 0;
    const char *mode = "both";

    int opt;
    while ((opt = getopt(argc, argv, "d:g:x:s:n:q:D:i:Hh")) != -1) {
        switch (opt) {
        case 'd':
            dev = optarg;
            break;
        case 'g':
            gpu = atoi(optarg);
            break;
        case 'x':
            gid_index = atoi(optarg);
            break;
        case 's':
            nbytes = (size_t)strtoull(optarg, NULL, 0);
            break;
        case 'n':
            iters = atoi(optarg);
            break;
        case 'q':
            nqps = atoi(optarg);
            break;
        case 'D':
            tx_depth = atoi(optarg);
            break;
        case 'i':
            port = atoi(optarg);
            break;
        case 'H':
            host_only = 1;
            break;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (optind < argc)
        mode = argv[optind];
    if (nqps < 1 || nqps > MAX_QPS)
        DIE("-q must be 1..%d", MAX_QPS);
    if (tx_depth < 1)
        DIE("-D tx_depth must be >= 1");
    if (iters < 1)
        DIE("-n iters must be >= 1");

    printf("======================================================\n");
    if (host_only)
        printf("some special machine types and scenarios RoCE loopback  (host DRAM <-> host DRAM)\n");
    else
        printf("some special machine types and scenarios XDR loopback  (xpu_malloc + ibverbs, no perftest)\n");
    printf("======================================================\n");
    printf("ibdev      : %s\n", dev);
    if (!host_only)
        printf("gpu / xpu  : %d\n", gpu);
    printf("gid index  : %d\n", gid_index);
    printf("port       : %d\n", port);
    printf("size       : %zu bytes\n", nbytes);
    printf("iters      : %d\n", iters);
    printf("qps        : %d\n", nqps);
    printf("tx_depth   : %d\n", tx_depth);
    printf("mode       : %s%s\n", mode, host_only ? "  (host-only)" : "");
    printf("BKCL_ENABLE_XDR        = %s\n", getenv("BKCL_ENABLE_XDR"));
    printf("CUDA_ENABLE_P2P_NO_UVA = %s\n", getenv("CUDA_ENABLE_P2P_NO_UVA"));
    printf("\n");

    void *host = NULL;
    void *remote = NULL;
    void *dev_ptr = NULL;
    struct xpu_api xpu;
    memset(&xpu, 0, sizeof(xpu));
    const char *remote_name;
    const char *remote_mr_name;

    if (posix_memalign(&host, 4096, nbytes))
        DIE("posix_memalign host: %s", strerror(errno));
    memset(host, 0, nbytes);

    if (host_only) {
        if (posix_memalign(&remote, 4096, nbytes))
            DIE("posix_memalign remote host: %s", strerror(errno));
        memset(remote, 0, nbytes);
        remote_name = "host DRAM";
        remote_mr_name = "remote-host-dram";
    } else {
        struct cuda_api cuda;
        load_xpu(&xpu);
        load_xmlir_cudart(&cuda);
        if (xpu.set_device(gpu) != 0)
            DIE("xpu_set_device(%d) failed", gpu);

        dev_ptr = xpu_alloc(&xpu, nbytes);
        uintptr_t daddr = (uintptr_t)dev_ptr;
        printf("xpu_malloc = %p  in_HBM=%s\n", dev_ptr,
               (daddr >= HBM_BEGIN && daddr < HBM_END) ? "yes" : "NO");
        if (xpu.xmemset)
            xpu.xmemset(dev_ptr, 0, (uint64_t)nbytes);

        remote = xdr_host_pointer(&cuda, dev_ptr);
        remote_name = "XPU HBM";
        remote_mr_name = "xpu-xdr-hostPointer";
    }

    struct rdma rd;
    rdma_open(&rd, dev, port, gid_index, nqps, tx_depth);
    rd.host_mr = reg_mr(rd.pd, host, nbytes, "host-dram");
    rd.xpu_mr = reg_mr(rd.pd, remote, nbytes, remote_mr_name);

    int do_write = strcmp(mode, "read") != 0;
    int do_read = strcmp(mode, "write") != 0;
    if (do_write)
        run_op(&rd, 1, host, remote, nbytes, iters, 0xA5, 1, remote_name);
    if (do_read)
        run_op(&rd, 0, host, remote, nbytes, iters, 0x5A, 1, remote_name);

    printf("\n======================================================\n");
    if (host_only)
        printf("ALL HOST<->HOST RoCE TESTS PASSED\n");
    else
        printf("ALL HOST<->XPU XDR TESTS PASSED\n");
    printf("======================================================\n");

    ibv_dereg_mr(rd.host_mr);
    ibv_dereg_mr(rd.xpu_mr);
    for (int i = 0; i < rd.nqps; i++) {
        ibv_destroy_qp(rd.init_qp[i]);
        ibv_destroy_qp(rd.tgt_qp[i]);
    }
    free(rd.init_qp);
    free(rd.tgt_qp);
    ibv_destroy_cq(rd.cq);
    ibv_dealloc_pd(rd.pd);
    ibv_close_device(rd.ctx);
    free(host);
    if (host_only)
        free(remote);
    else if (xpu.xfree && dev_ptr)
        xpu.xfree(dev_ptr);
    return 0;
}
#endif /* XDR_AS_LIB */
