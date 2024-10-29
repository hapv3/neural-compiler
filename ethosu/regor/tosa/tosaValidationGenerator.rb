#!/usr/bin/env ruby
#
# SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

require 'asciidoctor'
require 'digest'
require 'optparse'
require 'rexml'
REGOR_OP_NAMES = {
  'ARGMAX':   'OpType::ArgMax',
  'AVG_POOL2D': 'OpType::AvgPool',
  'CONV2D':   'OpType::Conv2D',
  'CONV3D':   'OpType::Conv3D',
  'DEPTHWISE_CONV2D': 'OpType::DepthwiseConv2D',
  'FULLY_CONNECTED': 'OpType::FullyConnected',
  'MATMUL':   'OpType::MatMul',
  'MAX_POOL2D': 'OpType::MaxPool',
  'TRANSPOSE_CONV2D':  'OpType::TransposeConv2D',
  'CLAMP':  'OpType::Clamp',
  'SIGMOID': 'OpType::Sigmoid',
  'TANH':   'OpType::Tanh',
  'ADD':    'OpType::Add',
  'ARITHMETIC_RIGHT_SHIFT': 'OpType::Asr',
  'BITWISE_AND': 'OpType::And',
  'BITWISE_OR': 'OpType::Or',
  'BITWISE_XOR': 'OpType::Xor',
  'INTDIV':    'OpType::Div',
  'LOGICAL_AND': 'OpType::LogicalAnd',
  'LOGICAL_LEFT_SHIFT': 'OpType::SHL',
  'LOGICAL_RIGHT_SHIFT': 'OpType::SHR',
  'LOGICAL_OR': 'OpType::LogicalOr',
  'LOGICAL_XOR': 'OpType::LogicalXor',
  'MAXIMUM': 'OpType::Maximum',
  'MINIMUM': 'OpType::Minimum',
  'MUL': 'OpType::Mul',
  'POW': 'OpType::Pow',
  'SUB': 'OpType::Sub',
  'TABLE': 'OpType::Table',
  'ABS': 'OpType::Abs',
  'BITWISE_NOT': 'OpType::Not',
  'CEIL': 'OpType::Ceil',
  'CLZ': 'OpType::CLZ',
  'EXP': 'OpType::Exp',
  'FLOOR': 'OpType::Floor',
  'LOG': 'OpType::Log',
  'LOGICAL_NOT': 'OpType::LogicalNot',
  'NEGATE':    'OpType::Neg',
  'RECIPROCAL': 'OpType::Reciprocal',
  'RSQRT': 'OpType::Rsqrt',
  'SELECT': 'OpType::Select',
  'EQUAL': 'OpType::Equal',
  'GREATER': 'OpType::Greater',
  'GREATER_EQUAL': 'OpType::GreaterEqual',
  'REDUCE_ANY':   'OpType::ReduceAny',
  'REDUCE_ALL':   'OpType::ReduceAll',
  'REDUCE_MAX':   'OpType::ReduceMax',
  'REDUCE_MIN':   'OpType::ReduceMin',
  'REDUCE_PRODUCT': 'OpType::ReduceProduct',
  'REDUCE_SUM': 'OpType::ReduceSum',
  'CONCAT':   'OpType::Concat',
  'PAD':      'OpType::Pad',
  'RESHAPE':  'OpType::Reshape',
  'REVERSE':  'OpType::Reverse',
  'SLICE':    'OpType::Slice',
  'TILE':     'OpType::Tile',
  'TRANSPOSE': 'OpType::Transpose',
  'GATHER':   'OpType::Gather',
  'SCATTER':  'OpType::Scatter',
  'RESIZE':   'OpType::Resize',
  'CAST':     'OpType::Cast',
  'RESCALE':  'OpType::Rescale',
  'IDENTITY': 'OpType::Identity',
  'COND_IF':  'OpType::If',
  'WHILE_LOOP': 'OpType::While',
  'CUSTOM': 'OpType::Custom',
    #'FFT2D',      'OpType::CurrentlyUnsupported',
    #'RFFT2D',     'OpType::CurrentlyUnsupported',
    #'ERF',        'OpType::CurrentlyUnsupported',
    #'DIM',        'OpType::CurrentlyUnsupported'},
}
def parse_options
  options = {profile: 'BI'}
  OptionParser.new do |opts|
    opts.banner = "Usage: tosaValidationGenerator [options]"
    opts.on('-s [ARG]', '--specification [ARG]', "Path to the TOSA Specification git.") do |v|
      options[:spec] = v
    end
    opts.on('-p [ARG]', '--profile [ARG]', "TOSA profile (BI|MI)") do |v|
      options[:profile] = v
    end
    opts.on('-h', '--help', 'Display this help') do
      puts opts
      exit
    end
  end.parse!
  if (options[:spec].nil?)
    abort("No specification path (-s/--specification option required) ")
  end
  if (!File.file?("%s/tosa.xml" % options[:spec]) || !File.file?("%s/tosa_spec.adoc" % options[:spec]))
    abort("No TOSA Specification found at %s" % options[:spec])
  end
  if (options[:profile] != 'BI')
    abort("Profile %s not supported." % options[:profile])
  end
  options
end

def indent(level)
  " "*4*level
end

class TosaValidator
  @operators
  @error_set
  @level_set
  @require_set
  @operator_checks
  @level_limits
  @levelName
  @profile
  def initialize(xml, adoc, profile, level_name)
    xml_operators = xml.root.get_elements('//operator')
    adoc_operators = (adoc.find_by id: '_operators').first
    @operators = []
    @error_set = Set[]
    @level_set = Set[]
    @require_set = Set[]
    @operator_checks = {}
    @operator_checks.default([])
    @specVersion = {major: 0, minor: 0, patch: 0, draft: false}
    @level_limits = {}
    @profile = profile
    @level_name = level_name
    specVersionNode = xml.root.get_elements('//version')[0]
    if (specVersionNode != nil)
      @specVersion[:major] = specVersionNode['major']
      @specVersion[:minor] = specVersionNode['minor']
      @specVersion[:patch] = specVersionNode['patch']
      @specVersion[:draft] = specVersionNode['draft'] != 'false'
    end
    levels_node = xml.root.get_elements('//levels')[0]
    levels_node.elements.each do |level|
      @level_limits[level['name']] = {max_rank: level['max_rank'], max_kernel: level['max_kernel'], max_stride: level['max_stride'], max_scale: level['max_scale'], max_log2_size: level['max_log2_size'], max_nesting: level['max_nesting']}
    end
    xml_operators.each do |xml_operator|
      op = parse_operator(xml_operator, adoc_operators)
      @operators.append(op) unless op.nil?
    end
  end

  def versioned_nametag
    specArgs = [@specVersion[:major], @specVersion[:minor], @specVersion[:patch], @specVersion[:draft] ? "_draft":"", @profile]
    nametag = "Version_%s_%s_%s%s_Profile_%s" % specArgs
  end

  def specArgs
    [@specVersion[:major], @specVersion[:minor], @specVersion[:patch], @specVersion[:draft] ? "draft":""]
  end

  def emit_file_header(should_modify: false, is_header: false)
    text = "//\n"
    text << "// SPDX-FileCopyrightText: Copyright 2023%s Arm Limited and/or its affiliates <open-source-office@arm.com>\n" % ((Time.now.year > 2023) ? "-" + Time.now.year.to_s : "")
    text << "//\n"
    text << "// SPDX-License-Identifier: Apache-2.0\n"
    text << "//\n"
    text << "// Licensed under the Apache License, Version 2.0 (the License); you may\n"
    text << "// not use this file except in compliance with the License.\n"
    text << "// You may obtain a copy of the License at\n"
    text << "//\n"
    text << "// www.apache.org/licenses/LICENSE-2.0\n"
    text << "//\n"
    text << "// Unless required by applicable law or agreed to in writing, software\n"
    text << "// distributed under the License is distributed on an AS IS BASIS, WITHOUT\n"
    text << "// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    text << "// See the License for the specific language governing permissions and\n"
    text << "// limitations under the License.\n"
    text << "//\n"
    if (should_modify)
      text << "// Partially generated by tosaValidationGenerator for TOSA Specification %u.%u.%u%s\n" % specArgs
      text << "// TODO: Implement the constraints.\n\n"
    else
      text << "// Automatically generated by tosaValidationGenerator for TOSA Specification %u.%u.%u%s\n" % specArgs
      text << "// Do not edit.\n\n"
    end
    if (is_header)
      text << "#pragma once\n\n"
    end
    text
  end

  def append_operator_check(check, operator)
    tmp = @operator_checks[check]
    if (tmp == nil)
      tmp = []
    end
    tmp.append(operator)
    @operator_checks[check] = tmp
  end
  def clean_constraint(constraint)
    constraint.gsub!(/\s+/, ' ')
    constraint.strip!
    if (constraint.delete_prefix!('('))
      constraint.delete_suffix!(')')
    end
    constraint
  end
  def parse_operator(xml_operator, adoc_operators)
    operation = Operation.new(xml_operator)
    adoc_operator = adoc_operators.find_by {|block| block.title == operation.name}.first
    source_blocks = adoc_operator.find_by style: 'source'
    error_checks = []
    level_checks = []
    require_checks = []
    argument_checks = []
    source_blocks.each {|block| error_checks += block.source.scan(/ERROR_IF(?<par>\((?:[^\(\)]|\g<par>)*\))/)}
    source_blocks.each {|block| level_checks += block.source.scan(/LEVEL_CHECK(?<par>\((?:[^\(\)]|\g<par>)*\))/)}
    source_blocks.each {|block| require_checks += block.source.scan(/REQUIRE(?<par>\((?:[^\(\)]|\g<par>)*\))/)}
    source_blocks.each {|block| argument_checks += block.source.scan(/(?<all>apply_broadcast(?<par>\((?:[^\(\)]|\g<par>)*\)))/)}
    error_checks.each do |a|
      constraint = clean_constraint(a[0])
      operation.error_checks.add(constraint)
      @error_set.add(constraint)
      append_operator_check(constraint, operation.name)
    end
    level_checks.each do |a|
      constraint = clean_constraint(a[0])
      operation.level_checks.add(constraint)
      @level_set.add(constraint)
      append_operator_check(constraint, operation.name)
    end
    require_checks.each do |a|
      constraint = clean_constraint(a[0])
      operation.require_checks.add(constraint)
      @require_set.add(constraint)
      append_operator_check(constraint, operation.name)
    end
    argument_checks.each do |a|
      constraint = clean_constraint(a[0])
      operation.error_checks.add(constraint)
      @error_set.add(constraint)
      append_operator_check(constraint, operation.name)
    end
    operation.shape_checks.each do |constraint|
      operation.error_checks.add(constraint)
      @error_set.add(constraint)
      append_operator_check(constraint, operation.name)
    end
   operation
  end

  def emit_check_header(check)
    header =  "\n{\n"
    line = indent(1) + "// Operators:"
    @operator_checks[check].each do |op|
      op_string = " %s," % op
      if ((line + op_string).length > 120)
        header << line + "\n"
        line = indent(1) + "// " + op_string
      else
        line << op_string
      end
    end
    header << line + "\n"
  end

  def emit_check(check, function_prefix, macro_name, is_header)
    function = "void %s_%s(const regor::Operation *op, %sconst Context &context)" % [function_prefix, name_tag(check), unused_tag(is_header)]
    if (is_header)
      function << ";\n"
    else
      function << emit_check_header(check)
      function << indent(1) + "static constexpr char constraint[] = \"%s(%s)\";\n" % [macro_name, check]
      function << indent(1) + "bool checkOk = true;\n"
      function << indent(1) + "checkOk = (op != nullptr);  // TODO: Implement check\n"
      function << indent(1) + "if ( !checkOk ) throw std::invalid_argument(constraint);\n"
      function << "}\n\n"
    end
  end

  def unused_tag(is_header)
    is_header ? "" : "[[maybe_unused]] "
  end

  def emit_argument_checks(is_header)
    body = ""
    if (is_header)
      body << "#include \"tosa/tosa_validator.hpp\"\n\n"
      body << "#include <map>\n"
      body << "#include <string>\n"
      body << "#include <string_view>\n"
      body << "#include <utility>\n"
      body << "#include <vector>\n"
      body << "namespace tosa\n{\n"
      body << "namespace validator\n{\n"
      body << "namespace checks\n{\n\n"
      body << "enum class Category\n{\n"
      body << indent(1) + "Input,\n"
      body << indent(1) + "Output,\n"
      body << indent(1) + "Attribute,\n"
      body << indent(1) + "ScalarAttribute\n"
      body << "};\n\n"
      body << "struct Argument\n{\n"
      body << indent(1) + "Category category;\n"
      body << indent(1) + "std::string name;\n"
      body << indent(1) + "std::string element_type;\n"
      body << indent(1) + "std::pair<int, int> rank_bounds = {-1, -1};\n"
      body << "};\n\n"
      body << "typedef std::map<std::string_view, std::string_view> Typesupport;\n\n"
      body << "void ValidateArguments(const regor::Operation *op, const std::vector<const Argument *> &arguments,\n"
      body << indent(1) + "const std::vector<Typesupport> &typesupports, const Context &context);\n\n"
    else
      body << "#include \"tosa/tosa_argument_checks.hpp\"\n\n"
      body << "#include <unordered_set>\n\n"
      body << "using namespace tosa::validator::checks;\n\n"
      body << "namespace\n{\n\n"
      body << "const std::unordered_set<std::string_view> tosaSupportedTypes = {\n"
      ["bool_t", "i4_t", "int4_t", "uint6_t", "i8_t", "int8_t", "uint8_t", "i16_t", "int16_t", "uint16_t", "i32_t", "int32_t", "i48_t", "int48_t", "fp16_t", "bf16_t", "fp32_t", "fp64_t", "index_t", "mode_t", "mul_t", "tensor_list_t", "tosa_graph_t"].each do |t|
        body << indent(1) + "\"%s\",\n" % t
      end
      body << "};\n\n"
      body << "bool CanResolveArgument(const Argument *argument, const Typesupport *typesupport)\n"
      body << "{\n"
      body << indent(1) + "if ( argument == nullptr ) return false;\n"
      body << indent(1) + "std::string_view typeName = argument->element_type;\n"
      body << indent(1) + "if ( typesupport )\n"
      body << indent(1) + "{\n"
      body << indent(2) + "if ( auto p = typesupport->find(typeName); p != typesupport->end() )\n"
      body << indent(2) + "{\n"
      body << indent(3) + "typeName = p->second;\n"
      body << indent(2) + "}\n"
      body << indent(1) + "}\n"
      body << indent(1) + "return tosaSupportedTypes.count(typeName) > 0;\n"
      body << "}\n\n"
      body << "bool CanResolveArguments(const std::vector<const Argument *> &arguments, const Typesupport *typesupport = nullptr)\n"
      body << "{\n"
      body << indent(1) + "for ( const auto argument : arguments )\n"
      body << indent(1) + "{\n"
      body << indent(2) + "if ( !CanResolveArgument(argument, typesupport) ) return false;\n"
      body << indent(1) + "}\n"
      body << indent(1) + "return true;\n"
      body << "}\n"
      body << "bool ArgumentsCanBeResolved(const std::vector<const Argument *> &arguments, const std::vector<Typesupport> &typesupports)\n"
      body << "{\n"
      body << indent(1) + "if ( CanResolveArguments(arguments) ) return true;\n"
      body << indent(1) + "for ( const auto &typesupport : typesupports )\n"
      body << indent(1) + "{\n"
      body << indent(2) + "if ( CanResolveArguments(arguments, &typesupport) ) return true;\n"
      body << indent(1) + "}\n"
      body << indent(1) + "return false;\n"
      body << "}\n"
      body << "}  // namespace\n"

      body << "namespace tosa\n{\n"
      body << "namespace validator\n{\n"
      body << "namespace checks\n{\n\n"
      body << "void ValidateArguments(const regor::Operation *, const std::vector<const Argument *> &arguments,\n"
      body << indent(1) + "const std::vector<Typesupport> &typesupports, const Context &)\n"
      body << "{\n"
      body << indent(1) + "if ( !ArgumentsCanBeResolved(arguments, typesupports) ) throw std::invalid_argument(\"Unsupported operation\");\n"
      body << indent(1) + "// TODO: Implement argument validation\n"
      body << "}\n\n"
    end
    body << "}  // namespace checks\n"
    body << "}  // namespace validator\n"
    body << "}  // namespace tosa\n"
  end
  def emit_version_validator_function
    validator = ""
    validator << "#include \"compiler/operation.hpp\"\n"
    validator << "#include \"tosa/tosa_argument_checks.hpp\"\n"
    validator << "#include \"tosa/tosa_error_checks.hpp\"\n"
    validator << "#include \"tosa/tosa_level_checks.hpp\"\n"
    validator << "#include \"tosa/tosa_require_checks.hpp\"\n"
    validator << "#include \"tosa/tosa_validator.hpp\"\n"
    validator << "using namespace tosa::validator;\n\n"
    validator << "using namespace tosa::validator::checks;\n\n"
    validator << "#define MAX_RANK (context.level == Level::Level8K ? 6 : (context.level == Level::Levelnone ? 32 : 0))\n\n"
    validator << "#define MAX_KERNEL (context.level == Level::Level8K ? 8192 : (context.level == Level::Levelnone ? 2147483647 : 0))\n\n"
    validator << "#define MAX_SCALE (context.level == Level::Level8K ? 256 : (context.level == Level::Levelnone ? 2048 : 0))\n\n"
    validator << "#define MAX_STRIDE (context.level == Level::Level8K ? 8192 : (context.level == Level::Levelnone ? 2147483647 : 0))\n\n"
    validator << "namespace\n{\n"
    @operators.each do |op|
      if (REGOR_OP_NAMES[op.name.to_sym]!=nil)
        validator << op.emit_validation_check
      end
    end
    validator << "}  // namespace\n"
    validator << "namespace tosa\n{\n"
    validator << "namespace validator\n{\n\n"
    validator << "void ValidateOperator_%s(const GraphApi::GraphOperation *graphOp, const Context &context)" % versioned_nametag
    validator << "\n{\n"
    validator << indent(1) + "const auto *op = static_cast<const regor::Operation *>(graphOp);\n"
    validator << indent(1) + "switch ( op->Type() )\n"
    validator << indent(1) + "{\n"
    @operators.each do |op|
      if (REGOR_OP_NAMES[op.name.to_sym])
        validator << indent(2) + "case regor::%s:\n" % REGOR_OP_NAMES[op.name.to_sym]
        validator << indent(3) + "ValidateOperator_%s(op, context);\n" % op.name
        validator << indent(3) + "break;\n" % op.name
      end
    end
    validator << indent(2) + "default:\n"
    validator << indent(3) + "throw std::invalid_argument(\"Unsupported operator\");\n"
    validator << indent(1) + "}\n"
    validator << "}\n\n"
    validator << "}  // namespace validator\n"
    validator << "}  // namespace tosa\n"
  end

  def update_checks(checks:, check_name:, function_prefix:, macro_name:)
    filename = "tosa_%s_checks.hpp" % check_name
    new_file = (!File.file?(filename))
    existing_checks =  new_file ? [] : File.readlines(filename).grep(/void #{function_prefix}_.*\(/)
    new_checks = []
    checks.each do |check|
      tag = name_tag(check)
      if (existing_checks.grep(/void #{function_prefix}_#{tag}\(/).empty?)
        new_checks.append(check)
      end
    end
    write_updated_checks(checks: new_checks, check_name: check_name, function_prefix: function_prefix, macro_name: macro_name, new_file: new_file, is_header: true)
    write_updated_checks(checks: new_checks, check_name: check_name, function_prefix: function_prefix, macro_name: macro_name, new_file: new_file, is_header: false)
  end

  def write_updated_checks(check_name:, function_prefix:, macro_name:, checks:, new_file:, is_header:)
    mode = new_file ? 'w' : 'a'
    file_name = "tosa_%s_checks.%s" % [check_name, (is_header ? 'hpp' : 'cpp')]
    File.open(file_name, mode) do |f|
      if (new_file)
        f.write emit_file_header(is_header: is_header, should_modify: !is_header)
        if (is_header)
          f.write "#include \"tosa/tosa_validator.hpp\"\n\n"
        else
        f.write "#include \"tosa_%s_checks.hpp\"\n\n" % check_name
        end
      end
      if (!checks.empty?)
        f.write "namespace tosa\n{\n"
        f.write "namespace validator\n{\n"
        f.write "namespace checks\n{\n"
        f.write "// Checks for TOSA Specification %u.%u.%u%s\n" % specArgs

        checks.each {|check| f.write  emit_check(check, function_prefix, macro_name, is_header)}
        f.write "}  // namespace checks\n"
        f.write "}  // namespace validator\n"
        f.write "}  // namespace tosa\n"
      end
    end
  end

  def update_argument_checks
    if (!File.file?("tosa_argument_checks.hpp"))
      File.open("tosa_argument_checks.hpp", "w"){|f| f.write emit_file_header(is_header: true) + emit_argument_checks(true)}
    end
    if (!File.file?("tosa_argument_checks.cpp"))  then
        File.open("tosa_argument_checks.cpp", "w"){|f| f.write emit_file_header(should_modify: true) + emit_argument_checks(false)}
    end
  end

  def update_error_checks
    update_checks(checks:@error_set, check_name: "error", function_prefix: "ErrorIfCheck", macro_name:"ERROR_IF")
  end

  def update_level_checks
    update_checks(checks:@level_set, check_name: "level", function_prefix: "LevelCheck", macro_name:"LEVEL_CHECK")
  end

  def update_require_checks
    update_checks(checks:@require_set, check_name: "require", function_prefix: "RequireCheck", macro_name:"REQUIRE")
  end

  def validator_function_def
    "void ValidateOperator_%s(const GraphApi::GraphOperation *graphOp, const Context &context)" % versioned_nametag
  end

  def supported_versions
    filename = "tosa_validator.hpp"
    new_file = (!File.file?(filename))
    versions = Set[validator_function_def + ";"]
    existing_versions = new_file ? [] : File.readlines(filename, chomp: true).grep(/void ValidateOperator_Version_.*_Profile/)
    versions.merge(existing_versions)
  end

  def write_validator_header(versions)
    File.open("tosa_validator.hpp", "w") do |f|
      f.write emit_file_header(is_header: true)

      f.write "#include \"include/graphapi.hpp\"\n\n"
      f.write "#include <functional>\n\n"
      f.write "#include <stdexcept>\n\n"
      f.write "namespace GraphApi\n{\n"
      f.write"struct GraphOperation;\n"
      f.write "}  // namespace GraphApi\n\n"
      f.write "namespace regor\n{\n"
      f.write "class Operation;\n"
      f.write "class Graph;\n"
      f.write "}  // namespace regor\n\n"
      f.write "namespace tosa\n{\n"
      f.write "namespace validator\n{\n\n"
      f.write "enum class Level\n{\n"
      @level_limits.each { |name, level| f.write indent(1) + "Level%s,\n" % name}
      f.write "};\n\n"
      f.write "struct Context\n{\n"
      f.write indent(1) + "uint32_t version = GraphApi::VERSION_TOSA_0_80;\n"
      f.write indent(1) + "int32_t profile = GraphApi::PROFILE_BASELINE;\n"
      f.write indent(1) + "Level level = Level::Level8K;\n"
      f.write indent(1) + "std::function<const regor::Graph*(const char *)> GetGraph;\n"
      f.write "};\n\n"
      versions.each { |version| f.write "%s\n" % version }
      f.write "void ValidateOperator(const GraphApi::GraphOperation *graphOp, const Context &context = Context{})"
      f.write ";\n\n"
      f.write "}  // namespace validator\n"
      f.write "}  // namespace tosa\n"
    end
  end

  def write_validator_implementation(versions)
    File.open("tosa_validator.cpp", "w") do |f|
      f.write emit_file_header(is_header: false)
      f.write "#include \"tosa/tosa_validator.hpp\"\n\n"
      f.write "#include \"compiler/operation.hpp\"\n\n"
      f.write "namespace tosa\n{\n"
      f.write "namespace validator\n{\n\n"
      f.write "void ValidateOperator(const GraphApi::GraphOperation *graphOp, const Context &context)\n"
      f.write "{\n"
      f.write indent(1) + "if ( graphOp == nullptr ) throw std::invalid_argument(\"No operation\");\n"
      versions.each { |version|  f.write version_validator_checked_call(version)}
      f.write indent(1) + "throw std::invalid_argument(\"TOSA version or profile not supported\");\n"
      f.write "}\n\n"
      f.write "}  // namespace validator\n"
      f.write "}  // namespace tosa\n"
    end
  end

  def version_validator_checked_call(validator_name)
    if (match = validator_name.match(/_Version_(\d+)_(\d+)_(\d+)(.*)Profile_(.*)\(/))
      major, minor, patch, draft, profile = match.captures
      patch_string = (patch.empty? || patch.to_i == 0 ? "" : "_" + patch)
      draft_string = draft == "_" ? "" : "_DRAFT"
      tosa_version_const = "GraphApi::VERSION_TOSA_%d_%02d%s%s" % [major.to_i, minor.to_i, patch_string, draft_string]
      if (profile == "BI")
        tosa_profile_const = "GraphApi::PROFILE_BASELINE"
      elsif (profile == "MAIN")
        tosa_profile_const = "GraphApi::PROFILE_MAIN"
      else return
      end
      check = indent(1) + "if ( (context.version & 0xFFFFFF00) == %s && context.profile == %s )\n" % [tosa_version_const, tosa_profile_const]
      check << indent(1) + "{\n"
      check << indent(2) + "ValidateOperator%s(graphOp, context);\n" % /(_Version_.*Profile_.*)\(/.match(validator_name)[1]
      check << indent(2) + "return;\n"
      check << indent(1) + "}\n"
    end
  end

  def update_validator_function
    File.open("tosa_validator_%s.cpp" % versioned_nametag.downcase, "w")  {|f| f.write emit_file_header + emit_version_validator_function}
    write_validator_header(supported_versions)
    write_validator_implementation(supported_versions)
  end
end

class Argument
  @name
  @category
  @description
  @type
  @shape
  @element_type
  @rank_bounds
  @dimensions
  attr_reader  :name, :dimensions, :shape

  def initialize(xml_argument)
    @name = xml_argument['name']
    @category = xml_argument['category']
    @description = xml_argument.text('./description')
    @type = xml_argument['type']
    @shape = xml_argument['shape']
    parse_dimensions
    @element_type = xml_argument['tensor-element-type']
    if (@element_type == nil && @type != nil)
      @element_type = @type.chomp('*')
    end
    rank_node = xml_argument.get_elements('rank').first
    if(rank_node != nil)
      @rank_bounds = {min: rank_node['min'], max:rank_node['max']}
    else
      if (@dimensions)
        @rank_bounds = {min: @dimensions.size, max:@dimensions.size,}
      end
    end
  end
  def parse_dimensions
    dim_match = /\[([^\]]*)\]/.match(@shape)
    if (dim_match)
      dim_string = dim_match[1]
      @dimensions = dim_string.split(',') if (!dim_string.empty?)
    end
  end
  def to_s
    s = @name + ";" + @category + ";" + @type + ";" + @element_type
    if (@rank_bounds != nil)
      s <<  ";" + @rank_bounds[:min] + ";" + @rank_bounds[:max]
    end
    s
  end
  def scalar?
    if (@rank_bounds == nil)
      false
    else
      @rank_bounds[:min] == '0' && @rank_bounds[:max] == '0'
    end
  end
  def category_string
    s = "Category::"
    if (@category == "input")
      s << "Input"
    elsif (@category == "output")
      s << "Output"
    elsif (scalar?)
      s << "ScalarAttribute"
    else
      s << "Attribute"
    end
  end

  def bounds_string
    if (scalar? || @rank_bounds == nil)
      ""
    else
      indent(2) + "{%s, %s}\n" % [@rank_bounds[:min], @rank_bounds[:max]]
    end
  end

  def valid_name
    if (@name == 'operator') #'operator' is reserved word in C++
      'operatorName'
    else
      @name
    end
  end

  def emit_definition
    definition = indent(1) + "const Argument %s = {\n" % valid_name
    definition << indent(2) + "%s,\n" % category_string
    definition << indent(2) + "\"%s\",\n" % @name
    definition << indent(2) + "\"%s\",\n" % @element_type
    definition << "%s" % bounds_string
    description = "/*%s %s*/\n" % [@description, scalar? ? "" : "shape="+@shape]
    description_words = description.split(' ')
    description_line = indent(1) + "};"
    description_words.each do |word|
      if (description_line.length() + 1 + word.length() > 120)
        definition << description_line + "\n"
        description_line = indent(2) + word
      else
        description_line << " " + word
      end
    end
    definition << description_line + "\n"
  end
end

class Operation
  @name
  @error_checks
  @level_checks
  @require_checks
  @shape_ckecks
  @arguments
  @typesupport
  attr_reader :name, :arguments, :shape_checks
  attr_accessor :error_checks, :level_checks, :require_checks

  def initialize(xml_operator)
    @name = xml_operator.get_elements('name').first.text
    @error_checks = Set[]
    @level_checks = Set[]
    @require_checks = Set[]
    @shape_checks = Set[]
    @typesupport = parse_types(xml_operator)
    @arguments = parse_arguments(xml_operator)
  end

  def shape_match(shape, shape2)
    return false if (!shape || !shape2)
    shape1 = shape.clone
    while (shape1.size > 0)
      dim = shape1.pop
      next if (Integer(dim, 10, exception: false))
      shape2.each { |dim2| return true if (dim == dim2) }
    end
    false
  end

  def add_argument_shape_checks(arguments)
    args = arguments.clone
    while (args.size > 0)
      arg = args.pop
      args.each do |arg1|
        if (shape_match(arg.dimensions, arg1.dimensions))
          @shape_checks.add("shapeCheck(%s, %s, %s, %s)" % [arg.name, arg.shape, arg1.name, arg1.shape])
        elsif (arg.shape == arg1.shape && arg.shape != "-" && !/\[([^\]]*)\]/.match(arg.shape))
          @shape_checks.add("rankCheck(%s, %s)" % [arg.name, arg1.name])
          x = 1
        end
      end
    end
  end

  def parse_arguments(xml_operator)
    arguments = []
    xml_arguments = xml_operator.get_elements('arguments')
    if (xml_arguments.size() > 0)
      xml_arguments[0].get_elements('argument').each{|a|
      arguments.append(Argument.new(a))}
    end
    add_argument_shape_checks(arguments)
    arguments
  end

  def parse_types(xml_operator)
    types = {}
    typenames = []
    xml_typesupports = xml_operator.get_elements('./typesupport')
    if (xml_typesupports.size() > 0)
      xml_types = xml_operator.get_elements('./types').first.get_elements('./type')
      xml_types.each {|xml_type| typenames.append(xml_type['name'])}
      xml_typesupports.each {|t| parse_typesupport(t, types, typenames)}
    end
    types
  end

  def parse_typesupport(xml_typesupport, types, typenames)
    if (xml_typesupport.children.size == 0)
      mode_types = {}
      mode = xml_typesupport['mode']
      typenames.each {|typename| mode_types[typename] = xml_typesupport[typename] }
      types[mode] = mode_types
    end
    types
  end

  def emit_validation_check()
    function = emit_validation_header()
    function << emit_argument_checks()
    function << emit_error_checks()
    function << emit_level_checks()
    function << emit_require_checks()
    function << emit_validation_footer()
  end

  def emit_validation_header()
    header = "void ValidateOperator_%s(const regor::Operation *op, const Context &context)\n" % @name
    header << "{\n"
  end

  def emit_argument_checks()
    checks = ""
    @arguments.each { |a| checks << a.emit_definition }
    checks << indent(1) + "const std::vector<const Argument *> arguments = {\n"
    @arguments.each { |a| checks << indent(2) + "&%s,\n" % a.valid_name}
    checks << indent(1) +"};\n"
    checks << indent(1) + "const std::vector<Typesupport> typesupports = {%s" % (@typesupport.empty? ? "" : "\n")
    @typesupport.each do |mode, types|
      checks << indent(2) + "{\n"
      types.each { |name, type| checks << indent(3) + "{\"%s\", \"%s\"},\n"% [name, type]}
      checks << indent(2) +"},  // %s\n" % mode
    end
    checks << (@typesupport.empty? ? "" : indent(1)) + "};\n"
    checks << indent(1) + "ValidateArguments(op, arguments, typesupports, context);\n"
  end

  def emit_error_checks()
    checks = ""
    @error_checks.each {|check| checks << indent(1) + "ErrorIfCheck_%s(op, context);\n" % name_tag(check)}
    checks
  end

  def emit_level_checks()
    checks = ""
    @level_checks.each {|check| checks << indent(1) + "LevelCheck_%s(op, context);\n" % name_tag(check)}
    checks
  end

  def emit_require_checks()
    checks = ""
    @require_checks.each {|check| checks << indent(1) + "RequireCheck_%s(op, context);\n" % name_tag(check)}
    checks
  end

  def emit_validation_footer()
    "}\n\n"
  end

end

def name_tag(obj)
  hashes = Digest::MD5.digest(obj.to_s).unpack('QQ')
  hash = (hashes[0] ^ hashes[1])
  tag = hash.to_s(36)
  if (hash < 0)
    tag[0] = 'N'
  end
  tag
end

options = parse_options
xmlfile = File.new("%s/tosa.xml" % options[:spec])
xml = REXML::Document.new(xmlfile)
doc = Asciidoctor.load_file "%s/tosa_spec.adoc" % options[:spec], safe: :safe, attributes: "generated=%s/out/gen pseudocode=%s/pseudocode" % [File.expand_path(options[:spec]), File.expand_path(options[:spec])]
validator = TosaValidator.new(xml, doc, 'BI', '8K')
validator.update_argument_checks
validator.update_error_checks
validator.update_level_checks
validator.update_require_checks
validator.update_validator_function
