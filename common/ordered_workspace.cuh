#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace mur {
// A deliberately narrow owner: all outstanding uses belong to last_stream().
// Stream handles and the CUDA context must outlive this object. No concurrent
// host mutation; parallel borrows must be joined before handing it back.
class OrderedWorkspace {
    void* pointer_{};
    cudaStream_t last_{};
    static void checked(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
public:
    struct Audit { int allocations=0, free_submissions=0, cleanup_errors=0; };
private:
    Audit* audit_; // Optional diagnostic owner must outlive this workspace.
public:
    OrderedWorkspace(size_t bytes,cudaStream_t stream,Audit* audit=nullptr):last_(stream),audit_(audit) {
        if(!bytes) throw std::invalid_argument("workspace must not be empty");
        checked(cudaMallocAsync(&pointer_,bytes,stream));
        if(audit_) ++audit_->allocations;
    }
    OrderedWorkspace(const OrderedWorkspace&)=delete;
    OrderedWorkspace& operator=(const OrderedWorkspace&)=delete;
    OrderedWorkspace(OrderedWorkspace&& other) noexcept:
        pointer_(std::exchange(other.pointer_,nullptr)),last_(other.last_),audit_(other.audit_) {}
    OrderedWorkspace& operator=(OrderedWorkspace&&)=delete;
    void* data() const noexcept {return pointer_;}
    cudaStream_t last_stream() const noexcept {return last_;}
    void handoff(cudaStream_t next) {
        if(next==last_) return;
        cudaEvent_t event{};checked(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
        auto status=cudaEventRecord(event,last_);
        if(status==cudaSuccess) status=cudaStreamWaitEvent(next,event);
        auto destroy_status=cudaEventDestroy(event);
        checked(status);
        // The wait has been accepted: update ownership even if event destruction
        // reports a prior asynchronous error. This method does not consume data.
        last_=next;checked(destroy_status);
    }
    void close() {
        if(!pointer_) return;
        checked(cudaFreeAsync(pointer_,last_));pointer_=nullptr;
        if(audit_) ++audit_->free_submissions;
    }
    ~OrderedWorkspace() noexcept {
        if(pointer_) {
            const auto status=cudaFreeAsync(pointer_,last_);
            if(audit_) {
                if(status==cudaSuccess) ++audit_->free_submissions;
                else ++audit_->cleanup_errors;
            }
        }
    }
};
} // namespace mur
