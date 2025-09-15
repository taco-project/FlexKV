#ifndef LOCK_FREE_Q_H
#define LOCK_FREE_Q_H

#include <atomic>
#include <memory>
#include <functional>

namespace flexkv {
    // Thread-safe and lock-free queue for worker threads
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        Node(const T& item) : data(item), next(nullptr) {}
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node(T{});
        head.store(dummy);
        tail.store(dummy);
    }
    
    ~LockFreeQueue() {
        Node* current = head.load();
        while (current) {
            Node* next = current->next.load();
            delete current;
            current = next;
        }
    }
    
    void push(const T& item) {
        Node* new_node = new Node(item);
        Node* old_tail = tail.load();
        
        while (true) {
            Node* next = old_tail->next.load();
            if (old_tail == tail.load()) {
                if (next == nullptr) {
                    if (old_tail->next.compare_exchange_weak(next, new_node)) {
                        tail.compare_exchange_strong(old_tail, new_node);
                        return;
                    }
                } else {
                    tail.compare_exchange_strong(old_tail, next);
                }
            }
        }
    }
    
    bool pop(T& item) {
        Node* old_head = head.load();
        Node* old_tail = tail.load();
        
        while (true) {
            Node* next = old_head->next.load();
            if (old_head == head.load()) {
                if (old_head == old_tail) {
                    if (next == nullptr) {
                        return false;
                    }
                    tail.compare_exchange_strong(old_tail, next);
                } else {
                    if (next == nullptr) {
                        return false;
                    }
                    item = next->data;
                    if (head.compare_exchange_weak(old_head, next)) {
                        delete old_head;
                        return true;
                    }
                }
            }
        }
    }

    // Traverse the queue and invoke callback for each node's data in FIFO order.
    // IMPORTANT: This is only safe when no concurrent pops are deleting nodes
    // (e.g., in a quiescent state or when external synchronization prevents node reclamation).
    template<typename Fn>
    void Traverse(Fn&& fn) const {
        Node* cur = head.load(std::memory_order_acquire);
        if (cur == nullptr) {
            return;
        }
        Node* node = cur->next.load(std::memory_order_acquire); // skip dummy
        while (node != nullptr) {
            fn(node->data);
            node = node->next.load(std::memory_order_acquire);
        }
    }
};

} // namespace flexkv

#endif // LOCK_FREE_Q_H