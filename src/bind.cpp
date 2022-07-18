#include <pybind11/functional.h>
#include "mapper.h"

namespace py = pybind11;

namespace torchrec_mapper {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Mapper>(m, "Mapper")
    .def(py::init<int64_t>())
    .def("map", &Mapper::Map);

  py::class_<Future>(m, "Future")
    .def("wait", &Future::Wait);
}

}  // namespace mapper
