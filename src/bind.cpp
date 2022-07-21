#include <pybind11/functional.h>
#include "mapper_collection.h"

namespace py = pybind11;

namespace torchrec_mapper {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MapperCollection>(m, "Mapper")
    .def(py::init<int64_t, int64_t>())
    .def("map", &MapperCollection::Map);

  py::class_<FutureCollection>(m, "Future")
    .def("wait", &FutureCollection::Wait);
}

}  // namespace mapper
