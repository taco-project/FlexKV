#pragma once
#include "gdr/runtime.h"
#include "gdr/transport.h"
#include "region_batch.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace flexkv::gdr {
namespace py = pybind11;
inline Config parse_config(py::dict value) {
  Config c;
  c.enabled = value["enabled"].cast<bool>();
  c.qp_count = value["qp_count"].cast<int>();
  c.use_dedicated_gdr_thread = value["use_dedicated_gdr_thread"].cast<bool>();
  c.nic_names = value["nic_names"].cast<std::vector<std::string>>();
  auto diag = value["diagnostics"].cast<py::dict>();
  c.summary = diag["summary"].cast<bool>();
  c.timing = diag["timing"].cast<bool>();
  c.probe = diag["probe"].cast<bool>();
  auto direction = [](py::dict d) {
    DirectionConfig out;
    const auto path = d["path"].cast<std::string>();
    if (path == "original") out.path = Path::Original;
    else if (path == "ce") out.path = Path::CE;
    else if (path == "kernel") out.path = Path::Kernel;
    else if (path == "gdr") out.path = Path::GDR;
    else if (path == "auto") out.path = Path::Auto;
    else throw std::invalid_argument("Invalid GDR path");
    auto selector = d["selector"].cast<std::string>();
    if (selector == "avgbytes") out.selector = Selector::AvgBytes;
    else if (selector == "segments") out.selector = Selector::Segments;
    else throw std::invalid_argument("Invalid GDR selector");
    auto failure = d["on_init_failure"].cast<std::string>();
    if (failure != "error" && failure != "original") throw std::invalid_argument("Invalid GDR fallback");
    out.fallback_on_init_failure = failure == "original";
    out.gdr_when_ce_avg_bytes_below = d["gdr_when_ce_avg_bytes_below"].cast<uint64_t>();
    out.gdr_max_block_fragment_bytes = d["gdr_max_block_fragment_bytes"].cast<uint64_t>();
    out.segment_threshold = d["segment_threshold"].cast<int>();
    return out;
  };
  c.h2d = direction(value["h2d"].cast<py::dict>());
  auto d2h = value["d2h"].cast<py::dict>();
  c.d2h = direction(d2h);
  c.read_depth = d2h["read_depth"].cast<int>();
  c.d2h_auto_layer_parallel = d2h["auto_layer_parallel"].cast<bool>();
  validate(c);
  return c;
}

inline py::class_<RegionBatchGroup> bind(py::module_ &m) {
  py::class_<Transport>(m, "GdrEndpoint")
      .def(py::init([](int device, int qps, const std::string &nic, int depth) {
        const c10::cuda::CUDAGuard guard(device);
        return std::make_unique<Transport>(device, qps, nic, false, "", depth);
      }), py::arg("device"), py::arg("qps"), py::arg("nic"), py::arg("read_depth") = 16)
      .def("export_tensor", [](Transport &self, const torch::Tensor &t) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "GDR owner requires contiguous CUDA tensor");
        const c10::cuda::CUDAGuard guard(t.device());
        self.register_device(t.storage().data_ptr().get(), t.storage().nbytes());
        size_t bytes = t.numel() * t.element_size();
        return py::make_tuple(self.gpu_address(t.data_ptr(), bytes), bytes, self.device_key(t.data_ptr(), bytes));
      }, py::keep_alive<1, 2>())
      .def_property_readonly("nic", &Transport::nic)
      .def_property_readonly("endpoint", &Transport::endpoint)
      .def_property_readonly("read_depth", &Transport::read_depth)
      .def("accept", &Transport::accept)
      .def("release", &Transport::release);
  m.def("gdr_compiled", &Transport::compiled);
  m.def("gdr_synchronize_owner", [](int device) {
    const c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "GDR owner producer fence failed");
  }, py::call_guard<py::gil_scoped_release>());
  auto cls = py::class_<RegionBatchGroup>(m, "RegionBatchGroup");
  cls.def("configure_gdr", [](RegionBatchGroup &self, py::dict value) {
        auto config = parse_config(value);
        py::gil_scoped_release release;
        self.configure_gdr(config);
      })
      .def("close_gdr", &RegionBatchGroup::close_gdr, py::call_guard<py::gil_scoped_release>())
      .def("gdr_statistics", &RegionBatchGroup::gdr_statistics)
      .def("gdr_register_local", [](RegionBatchGroup &self, int rank, const torch::Tensor &t) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "GDR local registration requires CUDA owner");
        self.gdr_context().register_local(rank, t.storage().data_ptr().get(), t.storage().nbytes());
      }, py::keep_alive<1, 3>())
      .def("gdr_register_host", [](RegionBatchGroup &self, int rank, const torch::Tensor &t) {
        TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "GDR host registration requires CPU tensor");
        self.gdr_context().register_host(rank, t.storage().data_ptr().get(), t.storage().nbytes());
      }, py::keep_alive<1, 3>())
      .def("gdr_prepare_remote", [](RegionBatchGroup &self, int rank, const std::string &nic, const std::string &endpoint) {
        return self.gdr_context().prepare_remote(rank, nic, endpoint);
      })
      .def("gdr_read_depth", [](RegionBatchGroup &self, int rank) { return self.gdr_context().read_depth(rank); })
      .def("gdr_connect_remote", [](RegionBatchGroup &self, int rank, const std::vector<uint32_t> &qpns, py::list mappings) {
        std::vector<RemoteRegion> regions;
        for (auto item : mappings) {
          auto t = py::cast<py::tuple>(item);
          if (t.size() != 4) throw std::invalid_argument("Invalid GDR region tuple");
          regions.push_back({t[0].cast<uintptr_t>(), t[1].cast<size_t>(), t[2].cast<uint64_t>(), t[3].cast<uint32_t>()});
        }
        self.gdr_context().connect_remote(rank, qpns, regions);
      })
      .def("gdr_fail_init", [](RegionBatchGroup &self, int rank, const std::string &reason) {
        self.gdr_context().fail_init(rank, reason);
      });
  return cls;
}
} // namespace flexkv::gdr
