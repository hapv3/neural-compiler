//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "include/regor_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "include/regor.h"

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

namespace py = pybind11;

struct PyRegorMemoryAccess
{
    std::string accessType;
    int64_t bytesRead = 0;
    int64_t bytesWritten = 0;
    int64_t accessCycles = 0;

    std::string ToString() const
    {
        std::string ret;
        ret += "Access Type[" + accessType + "]\n";
        ret += "\tread  = " + std::to_string(bytesRead) + "\n";
        ret += "\twrite = " + std::to_string(bytesWritten) + "\n";
        ret += "\tcycles = " + std::to_string(accessCycles) + "\n";
        return ret;
    }
};

struct PyRegorMemoryPerf
{
    std::string memoryName;
    int64_t peakUsage = 0;
    std::unordered_map<std::string, PyRegorMemoryAccess> accesses;

    std::string ToString() const
    {
        std::string ret;
        ret += "Memory[" + memoryName + "]\n";
        ret += "\tPeak usage  = " + std::to_string(peakUsage) + "\n";
        return ret;
    }
};

struct PyRegorPerfReport
{
    int64_t npuCycles = 0;
    int64_t cpuCycles = 0;
    int64_t totalCycles = 0;
    int64_t macCount = 0;
    int64_t cpuOps = 0;
    int64_t npuOps = 0;
    int64_t cascadedOps = 0;
    int64_t cascades = 0;
    int64_t originalWeights = 0;
    int64_t encodedWeights = 0;
    std::string stagingMemoryArea = "";

    std::unordered_map<std::string, PyRegorMemoryPerf> memories;

    std::string ToString() const
    {
        assert(npuOps >= 0);
        assert(cpuOps >= 0);
        assert(cascadedOps >= 0);
        assert(cascades >= 0);

        assert(npuOps <= (std::numeric_limits<int64_t>::max() - cpuOps));
        auto totalOpsBc = npuOps + cpuOps;
        assert(totalOpsBc >= cascadedOps);
        assert((totalOpsBc - cascadedOps) <= (std::numeric_limits<int64_t>::max() - cascades));

        std::string ret;
        ret += "NPU Cycles = " + std::to_string(npuCycles) + "\n";
        ret += "CPU Cycles = " + std::to_string(cpuCycles) + "\n";
        ret += "Total Cycles = " + std::to_string(totalCycles) + "\n";
        ret += "Total MACs = " + std::to_string(macCount) + "\n";
        ret += "CPU Operations = " + std::to_string(cpuOps) + "\n";
        ret += "NPU Operations = " + std::to_string(npuOps) + "\n";
        ret += "Total operations (before cascading) = " + std::to_string(totalOpsBc) + "\n";
        ret += "Total operations (after cascading) = " + std::to_string(totalOpsBc - cascadedOps + cascades) + "\n";
        ret += "Original Weights = " + std::to_string(originalWeights) + "\n";
        ret += "Encoded Weights = " + std::to_string(encodedWeights) + "\n";
        for ( const auto &[memName, memory] : memories )
        {
            ret += memory.ToString();
            for ( const auto &[_memName, access] : memory.accesses )
            {
                ret += access.ToString();
            }
        }
        return ret;
    }
};

struct PyRegorDatabaseTable
{
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> data;
};

struct PyRegorDatabase
{
    std::unordered_map<std::string, PyRegorDatabaseTable> tables;
};

struct PyRegorCompiledModel
{
    PyRegorCompiledModel() {}

    void SetPerfReport(PyRegorPerfReport &&report) { perf_report = std::move(report); }
    void SetOptDatabase(PyRegorDatabase &&database) { opt_database = std::move(database); }

    PyRegorPerfReport perf_report;
    PyRegorDatabase opt_database;
};

struct PyRegorCompiledRawModelConstantTensor
{
    PyRegorCompiledRawModelConstantTensor() = default;
    PyRegorCompiledRawModelConstantTensor(uint8_t region_, py::bytes data_) : region(region_), data(std::move(data_)) {}

    uint8_t region = 0;
    py::bytes data;
};

struct PyRegorCompiledRawModelNonConstantTensor
{
    PyRegorCompiledRawModelNonConstantTensor() = default;
    PyRegorCompiledRawModelNonConstantTensor(int64_t region_, int64_t address_, uint32_t size_, uint8_t element_size_, std::vector<uint32_t> &shape_) :
            region(region_), address(address_), size(size_), element_size(element_size_), shape(shape_)
    {
    }

    uint8_t region = 0;
    uint64_t address = 0;
    uint32_t size = 0;
    uint8_t element_size = 0;
    std::vector<uint32_t> shape;
};

struct PyRegorCompiledRawModel : PyRegorCompiledModel
{
    PyRegorCompiledRawModel() : PyRegorCompiledModel() {}

    py::bytes command_stream;
    PyRegorCompiledRawModelConstantTensor read_only;
    PyRegorCompiledRawModelNonConstantTensor scratch;
    PyRegorCompiledRawModelNonConstantTensor scratch_fast;
    std::vector<PyRegorCompiledRawModelNonConstantTensor> inputs;
    std::vector<PyRegorCompiledRawModelNonConstantTensor> outputs;
    std::vector<PyRegorCompiledRawModelNonConstantTensor> variables;
};

struct PyRegorCompiledTFLiteModel : PyRegorCompiledModel
{
    PyRegorCompiledTFLiteModel() : PyRegorCompiledModel(), model(py::none()) {}
    PyRegorCompiledTFLiteModel(py::object model_) : PyRegorCompiledModel(), model(std::move(model_)) {}

    py::object model;
};

class PyRegor
{
public:
    PyRegor(const std::string &arch, bool verbose)
    {
        if ( !regor_create(&_context, arch.c_str()) )
        {
            throw std::invalid_argument("Unknown architecture " + arch);
        }
        std::cout.setf(std::ios::unitbuf);
        std::cout.flush();
        regor_set_logging(&PyRegor::Log, verbose ? ~0u : 0u);
    }

    ~PyRegor() { regor_destroy(_context); }

    void SetSystemConfig(const std::string &config)
    {
        if ( !regor_set_system_config(_context, config.c_str(), config.size()) )
        {
            throw std::invalid_argument("Invalid System Config");
        }
    }

    void SetCompilerOptions(const std::string &options)
    {
        if ( !regor_set_compiler_options(_context, options.c_str(), options.size()) )
        {
            throw std::invalid_argument("Invalid Compiler Options");
        }
    }

    py::object PyCompile(const py::buffer &input, const std::string &fmt)
    {
        // Extract input buffer and size of input buffer
        py::buffer_info info(input.request());
        const void *in_data = static_cast<const void *>(info.ptr);
        size_t in_size = size_t(std::max<py::ssize_t>(info.size, 0));

        // Compile the input buffer and return a subclass of PyRegorCompiledModel
        return Compile(in_data, in_size, fmt);
    }

    PyRegorPerfReport GetPerfReport()
    {
        PyRegorPerfReport pyPerf;
        regor_perf_report_t report;
        int rStatus;
        (void)(rStatus);
        report.access = nullptr;

        rStatus = regor_get_perf_report(_context, &report);
        assert(rStatus);
        // Parse accessCount and call again to populate access
        std::vector<regor_memory_access_perf_t> access;
        if ( report.accessCount > 0 )
        {
            access.resize(report.accessCount);
        }

        report.access = access.data();
        rStatus = regor_get_perf_report(_context, &report);
        assert(rStatus);

        pyPerf.npuCycles = report.npuCycles;
        pyPerf.cpuCycles = report.cpuCycles;
        pyPerf.totalCycles = report.totalCycles;
        pyPerf.macCount = report.macCount;
        pyPerf.cpuOps = report.cpuOps;
        pyPerf.npuOps = report.npuOps;
        pyPerf.cascadedOps = report.cascadedOps;
        pyPerf.cascades = report.cascades;
        pyPerf.originalWeights = report.originalWeights;
        pyPerf.encodedWeights = report.encodedWeights;
        if ( report.npuOps > 0 )
        {
            assert(report.numMemories <= int(std::size(report.peakUsages)));
            assert(report.stagingMemory >= 0 && report.stagingMemory < report.numMemories);
            pyPerf.stagingMemoryArea = report.peakUsages[report.stagingMemory].memoryName;
            for ( int i = 0; i < report.numMemories; ++i )
            {
                std::string name = report.peakUsages[i].memoryName;
                pyPerf.memories[name].memoryName = report.peakUsages[i].memoryName;
                pyPerf.memories[name].peakUsage = report.peakUsages[i].peakUsage;
            }
            for ( const auto &acc : access )
            {
                std::string name = acc.memoryName;
                pyPerf.memories[name].accesses[acc.accessType] = {acc.accessType, acc.bytesRead, acc.bytesWritten, acc.accessCycles};
            }
        }
        return pyPerf;
    }

    PyRegorDatabase GetOptDatabase()
    {
        regor::IRegorReporting *reporting = regor_get_reporting_interface(_context);
        assert(reporting);
        regor::IDatabase *db = reporting->OptimiserDatabase();
        return ToPyDatabase(db);
    }

private:
    static void Log(const void *data, size_t size)
    {
        assert(size <= std::numeric_limits<std::streamsize>::max());
        std::cout.write(reinterpret_cast<const char *>(data), size);
    }

    PyRegorDatabase ToPyDatabase(regor::IDatabase *db)
    {
        PyRegorDatabase pyDb;
        if ( db != nullptr )
        {
            regor::ITableIterator *table = db->Tables();
            while ( table->Next() )
            {
                // table name
                std::string name = table->Name();
                PyRegorDatabaseTable pyDbTable;

                // header
                regor::IRowIterator<std::string> *row = table->ColumnNames();
                bool isIndexed = (row->Id() > 0);
                if ( isIndexed ) pyDbTable.header.push_back("id");
                while ( row->Next() )
                {
                    pyDbTable.header.push_back(row->Value());
                }
                row->Release();

                // data
                int rowCount = table->Rows();
                for ( int i = 0; i < rowCount; i++ )
                {
                    std::vector<std::string> data;
                    row = table->Row(i);
                    if ( isIndexed ) data.push_back(std::to_string(row->Id()));
                    while ( row->Next() )
                    {
                        data.push_back(row->Value());
                    }
                    row->Release();
                    pyDbTable.data.push_back(data);
                }
                pyDb.tables[name] = std::move(pyDbTable);
            }
            table->Release();
        }
        return pyDb;
    }

    // Internal helper
    py::object Compile(const void *input, size_t in_size, const std::string &_fmt)
    {
        int rStatus;
        (void)(rStatus);

        // Capture these
        rStatus = regor_set_callback_arg(_context, this);
        assert(rStatus);

        regor_format_t fmt =
            _fmt == "TFLITE" ? REGOR_INPUTFORMAT_TFLITE :
            _fmt == "TOSA"   ? REGOR_INPUTFORMAT_TOSA :
                               REGOR_INPUTFORMAT_GRAPHAPI;

        if ( !regor_compile(_context, fmt, input, in_size, nullptr) )
        {
            std::string last_error;
            size_t last_error_len = 0;

            // Measure error length
            if ( !regor_get_error(_context, nullptr, &last_error_len) )
            {
                throw std::runtime_error("Compilation failed: Failed to fetch error");
            }
            last_error.resize(last_error_len);

            if ( !regor_get_error(_context, last_error.data(), &last_error_len) )
            {
                throw std::runtime_error("Compilation failed: Failed to fetch error");
            }
            else
            {
                throw std::runtime_error("Compilation failed: " + last_error);
            }
        }

        // Get all compiler output
        std::vector<regor::IRegorBlob *> blobs;
        {
            regor::IRegorBlob *blob;
            while ( regor_get_output(_context, &blob) )
            {
                if ( blob ) blobs.push_back(blob);
            }
        }

        if ( blobs.size() == 1 )
        {
            // Likely TFLite output

            PyRegorCompiledTFLiteModel tfl;
            tfl.SetPerfReport(GetPerfReport());
            tfl.SetOptDatabase(GetOptDatabase());

            int64_t size;
            void *buf = blobs[0]->Map(size);
            assert(size >= 0 && size <= int64_t(std::numeric_limits<py::ssize_t>::max()));
            tfl.model = py::bytes(reinterpret_cast<char *>(buf), py::ssize_t(size));
            blobs[0]->Unmap(buf);
            blobs[0]->Release();

            return py::cast(tfl);
        }
        else if ( blobs.size() > 1 )
        {
            // Likely raw output

            PyRegorCompiledRawModel raw;
            raw.SetPerfReport(GetPerfReport());
            raw.SetOptDatabase(GetOptDatabase());

            for ( auto &blob : blobs )
            {
                int64_t size;
                char *buf = reinterpret_cast<char *>(blob->Map(size));

                regor_raw_tensor_header_t header;
                std::copy_n(buf, sizeof(header), reinterpret_cast<char *>(&header));

                char *data;
                uint32_t data_size;
                uint8_t region;
                uint64_t address;
                uint8_t element_size;
                std::vector<uint32_t> shape;

                switch ( header.type )
                {
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_COMMAND_STREAM:
                        data = buf + sizeof(header);
                        data_size = header.tensor.command_stream.size;
                        raw.command_stream = py::bytes(data, data_size);
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_READ_ONLY:
                        data = buf + sizeof(header);
                        data_size = header.tensor.read_only.size;
                        raw.read_only.region = header.tensor.read_only.region;
                        raw.read_only.data = py::bytes(data, data_size);
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH:
                        raw.scratch.region = header.tensor.scratch.region;
                        raw.scratch.size = header.tensor.scratch.size;
                        raw.scratch.address = header.tensor.scratch.address;
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH_FAST:
                        raw.scratch_fast.region = header.tensor.scratch_fast.region;
                        raw.scratch_fast.size = header.tensor.scratch_fast.size;
                        raw.scratch_fast.address = header.tensor.scratch_fast.address;
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_INPUT:
                        region = header.tensor.input.region;
                        address = header.tensor.input.address;
                        data_size = header.tensor.input.size;
                        element_size = header.tensor.input.element_size;
                        shape.insert(shape.end(), header.tensor.input.shape, header.tensor.input.shape + 4);
                        raw.inputs.emplace_back(region, address, size, element_size, shape);
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_OUTPUT:
                        region = header.tensor.output.region;
                        address = header.tensor.output.address;
                        data_size = header.tensor.output.size;
                        element_size = header.tensor.output.element_size;
                        shape.insert(shape.end(), header.tensor.output.shape, header.tensor.output.shape + 4);
                        raw.outputs.emplace_back(region, address, size, element_size, shape);
                        break;
                    case regor_raw_tensor_header_t::RAW_TENSOR_TYPE_VARIABLE:
                        region = header.tensor.variable.region;
                        address = header.tensor.variable.address;
                        data_size = header.tensor.variable.size;
                        element_size = header.tensor.variable.element_size;
                        shape.insert(shape.end(), header.tensor.variable.shape, header.tensor.variable.shape + 4);
                        raw.variables.emplace_back(region, address, size, element_size, shape);
                        break;
                    default:
                        break;
                }

                blob->Unmap(buf);
                blob->Release();
            }

            return py::cast(raw);
        }
        else
        {
            throw std::runtime_error("Compilation generated no output blobs");
        }
    }

    regor_context_t _context;
};

PYBIND11_MODULE(regor, m)
{
    m.doc() = R"pbdoc(
        Welcome to Regor - the brightest star in Vela
    )pbdoc";

    m.attr("__version__") = STRINGIFY(REGOR_VERSION);

    py::class_<PyRegorMemoryAccess>(m, "MemoryAccess", "Regor memory accesses")
        .def(py::init<>())
        .def_readwrite("accessType", &PyRegorMemoryAccess::accessType, "Access type")
        .def_readwrite("bytesRead", &PyRegorMemoryAccess::bytesRead, "Bytes read")
        .def_readwrite("bytesWritten", &PyRegorMemoryAccess::bytesWritten, "Bytes written")
        .def_readwrite("accessCycles", &PyRegorMemoryAccess::accessCycles, "Total access cycles")
        .def("__repr__", &PyRegorMemoryAccess::ToString);
    py::class_<PyRegorMemoryPerf>(m, "MemoryPerf", "A Regor memory performance report")
        .def(py::init<>())
        .def_readwrite("memoryName", &PyRegorMemoryPerf::memoryName, "Memory name")
        .def_readwrite("peakUsage", &PyRegorMemoryPerf::peakUsage, "Peak usage")
        .def_readwrite("accesses", &PyRegorMemoryPerf::accesses, "Accesses")
        .def("__repr__", &PyRegorMemoryPerf::ToString);
    py::class_<PyRegorPerfReport>(m, "PerfReport", "A Regor performance report")
        .def(py::init<>())
        .def_readwrite("npuCycles", &PyRegorPerfReport::npuCycles, "NPU elapsed cycles")
        .def_readwrite("cpuCycles", &PyRegorPerfReport::cpuCycles, "CPU elapsed cycles")
        .def_readwrite("totalCycles", &PyRegorPerfReport::totalCycles, "Total elapsed time in cycles")
        .def_readwrite("macCount", &PyRegorPerfReport::macCount, "Number of Multiply-Accumulate operations")
        .def_readwrite("cpuOps", &PyRegorPerfReport::cpuOps, "Number of CPU operations")
        .def_readwrite("npuOps", &PyRegorPerfReport::npuOps, "Number of NPU operations")
        .def_readwrite("cascadedOps", &PyRegorPerfReport::cascadedOps, "Number of cascaded operations")
        .def_readwrite("cascades", &PyRegorPerfReport::cascades, "Number of cascades")
        .def_readwrite("originalWeights", &PyRegorPerfReport::originalWeights, "Weights size (uncompressed)")
        .def_readwrite("encodedWeights", &PyRegorPerfReport::encodedWeights, "Weights size (compressed)")
        .def_readwrite("stagingMemoryArea", &PyRegorPerfReport::stagingMemoryArea, "Staging memory area")
        .def_readwrite("memories", &PyRegorPerfReport::memories, "Memory performance report")
        .def("__repr__", &PyRegorPerfReport::ToString);

    py::class_<PyRegorDatabaseTable>(m, "DatabaseTable", "A regor database table")
        .def(py::init<>())
        .def_readwrite("header", &PyRegorDatabaseTable::header, "database headers")
        .def_readwrite("data", &PyRegorDatabaseTable::data, "database data");

    py::class_<PyRegorDatabase>(m, "Database", "A regor database").def(py::init<>()).def_readwrite("tables", &PyRegorDatabase::tables, "database tables");

    py::class_<PyRegor>(m, "Regor", "The main Regor compiler class")
        .def(py::init<const std::string &, bool>())
        .def("SetSystemConfig", &PyRegor::SetSystemConfig, "Set the system configuration")
        .def("SetCompilerOptions", &PyRegor::SetCompilerOptions, "Set compiler options")
        .def("Compile", &PyRegor::PyCompile, "Compile the input model into a TFLite Flatbuffer", py::arg("input"), py::arg("fmt"))
        .def("GetPerfReport", &PyRegor::GetPerfReport, "Get the performance report for the latest compiled model")
        .def("GetOptDatabase", &PyRegor::GetOptDatabase, "Get the optimiser database for the latest compiled model");

    py::class_<PyRegorCompiledRawModelNonConstantTensor>(m, "CompiledRawModelNonConstantTensor", "A non-constant tensor of a Regor-compiled model in raw format")
        .def(py::init<>())
        .def_readwrite("region", &PyRegorCompiledRawModelNonConstantTensor::region, "The tensor's region")
        .def_readwrite("address", &PyRegorCompiledRawModelNonConstantTensor::address, "The tensor's address")
        .def_readwrite("size", &PyRegorCompiledRawModelNonConstantTensor::size, "The tensor's size")
        .def_readwrite("element_size", &PyRegorCompiledRawModelNonConstantTensor::element_size, "The tensor's element size")
        .def_readwrite("shape", &PyRegorCompiledRawModelNonConstantTensor::shape, "The tensor's shape");

    py::class_<PyRegorCompiledRawModelConstantTensor>(m, "CompiledRawModelConstantTensor", "A constant tensor of a Regor-compiled model in raw format")
        .def(py::init<>())
        .def_readwrite("region", &PyRegorCompiledRawModelConstantTensor::region, "The tensor's region")
        .def_readwrite("data", &PyRegorCompiledRawModelConstantTensor::data, "The tensor's constant data");

    py::class_<PyRegorCompiledModel>(m, "CompiledModel", "A Regor-compiled model")
        .def(py::init<>())
        .def_readwrite("perf_report", &PyRegorCompiledTFLiteModel::perf_report, "The performance report for the compiled model")
        .def_readwrite("opt_database", &PyRegorCompiledTFLiteModel::opt_database, "The optimiser database for the compiled model");

    py::class_<PyRegorCompiledRawModel, PyRegorCompiledModel>(m, "CompiledRawModel", "A Regor-compiled model in raw format")
        .def(py::init<>())
        .def_readwrite("command_stream", &PyRegorCompiledRawModel::command_stream, "The compiled model command stream")
        .def_readwrite("read_only", &PyRegorCompiledRawModel::read_only, "The compiled model weights")
        .def_readwrite("scratch", &PyRegorCompiledRawModel::scratch, "The compiled model scratch area")
        .def_readwrite("scratch_fast", &PyRegorCompiledRawModel::scratch_fast, "The compiled model scratch fast area")
        .def_readwrite("inputs", &PyRegorCompiledRawModel::inputs, "The compiled model inputs")
        .def_readwrite("outputs", &PyRegorCompiledRawModel::outputs, "The compiled model outputs")
        .def_readwrite("variables", &PyRegorCompiledRawModel::variables, "The compiled model variables");

    py::class_<PyRegorCompiledTFLiteModel, PyRegorCompiledModel>(m, "CompiledTFLiteModel", "A Regor-compiled TFLite model")
        .def(py::init<>())
        .def_readwrite("model", &PyRegorCompiledTFLiteModel::model, "The compiled model TFLite blob");

    m.def(
        "compile",
        [](const std::string &arch, py::buffer input, const std::string &fmt, const std::string &sysconfig,
            const std::string &options = "", bool verbose = false) -> py::object
        {
            PyRegor pyr(arch, verbose);
            pyr.SetSystemConfig(sysconfig);
            if ( options.size() > 0 )
            {
                pyr.SetCompilerOptions(options);
            }

            return pyr.PyCompile(input, fmt);
        },
        R"pbdoc(
            Compile a model
            Returns a compiled model
        )pbdoc",
        py::arg("arch"), py::arg("input"), py::arg("fmt"), py::arg("sysconfig"), py::arg("options") = "", py::arg("verbose") = false);
}
