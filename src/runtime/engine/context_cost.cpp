#include "runtime/engine/context_cost.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace ninfer::runtime {

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost();
const ContextPrefillCost& generic_context_prefill_cost();
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults();

namespace {

using Json = nlohmann::json;
using U128 = unsigned __int128;

constexpr std::size_t direction_index(ContextTransferDirection direction) noexcept {
    return static_cast<std::size_t>(direction);
}

std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

std::uint64_t saturating_product(std::uint64_t left, std::uint64_t right) noexcept {
    const U128 product = static_cast<U128>(left) * right;
    return product > std::numeric_limits<std::uint64_t>::max()
               ? std::numeric_limits<std::uint64_t>::max()
               : static_cast<std::uint64_t>(product);
}

std::uint64_t q32_product_ns(std::uint64_t coefficient, std::uint64_t units) noexcept {
    if (coefficient == 0 || units == 0) { return 0; }
    const U128 product        = static_cast<U128>(coefficient) * units;
    const U128 maximum_scaled = static_cast<U128>(std::numeric_limits<std::uint64_t>::max()) << 32U;
    if (product >= maximum_scaled) { return std::numeric_limits<std::uint64_t>::max(); }
    return static_cast<std::uint64_t>((product + kContextCostQ32One - 1U) >> 32U);
}

void require_object(const Json& value, std::string_view context) {
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be an object");
    }
}

void require_exact_members(const Json& value, std::initializer_list<std::string_view> required,
                           std::initializer_list<std::string_view> optional,
                           std::string_view context) {
    require_object(value, context);
    for (const std::string_view member : required) {
        if (!value.contains(std::string(member))) {
            throw std::invalid_argument(std::string(context) + " is missing '" +
                                        std::string(member) + "'");
        }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        const std::string_view member = it.key();
        const bool known = std::find(required.begin(), required.end(), member) != required.end() ||
                           std::find(optional.begin(), optional.end(), member) != optional.end();
        if (!known) {
            throw std::invalid_argument(std::string(context) + " contains unknown member '" +
                                        std::string(member) + "'");
        }
    }
}

std::uint64_t require_u64(const Json& value, std::string_view context) {
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    if (value.is_number_integer()) {
        const std::int64_t parsed = value.get<std::int64_t>();
        if (parsed >= 0) { return static_cast<std::uint64_t>(parsed); }
    }
    throw std::invalid_argument(std::string(context) + " must be an unsigned integer");
}

std::string require_nonempty_string(const Json& value, std::string_view context) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(std::string(context) + " must be a nonempty string");
    }
    return value.get<std::string>();
}

void validate_transfer(const std::array<ContextTransferCost, 3>& transfer,
                       std::string_view context) {
    for (std::size_t index = 0; index < transfer.size(); ++index) {
        const ContextTransferCost& value = transfer[index];
        if (value.batch_ns == 0 && value.operation_ns == 0 && value.ns_per_byte_q32 == 0) {
            throw std::invalid_argument(std::string(context) + "[" + std::to_string(index) +
                                        "] has no measurable cost");
        }
    }
}

void validate_prefill(const ContextPrefillCost& prefill, std::string_view context) {
    if (prefill.chunk_ns == 0 && prefill.token_ns_q32 == 0 && prefill.attention_pair_ns_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no text cost");
    }
    if (prefill.vision_item_ns == 0 && prefill.vision_patch_ns_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no Vision cost");
    }
}

ContextTransferCost parse_transfer_direction(const Json& value, std::string_view context) {
    require_exact_members(value, {"batch_ns", "operation_ns", "ns_per_byte_q32"}, {}, context);
    ContextTransferCost result{
        .batch_ns = require_u64(value.at("batch_ns"), std::string(context) + ".batch_ns"),
        .operation_ns =
            require_u64(value.at("operation_ns"), std::string(context) + ".operation_ns"),
        .ns_per_byte_q32 =
            require_u64(value.at("ns_per_byte_q32"), std::string(context) + ".ns_per_byte_q32"),
    };
    if (result.batch_ns == 0 && result.operation_ns == 0 && result.ns_per_byte_q32 == 0) {
        throw std::invalid_argument(std::string(context) + " has no measurable cost");
    }
    return result;
}

std::array<ContextTransferCost, 3> parse_transfer(const Json& value, std::string_view context) {
    require_exact_members(value, {"d2h", "h2d", "d2d"}, {}, context);
    return {
        parse_transfer_direction(value.at("d2h"), std::string(context) + ".d2h"),
        parse_transfer_direction(value.at("h2d"), std::string(context) + ".h2d"),
        parse_transfer_direction(value.at("d2d"), std::string(context) + ".d2d"),
    };
}

ContextPrefillCost parse_prefill(const Json& value, std::string_view context) {
    require_exact_members(value,
                          {"chunk_ns", "token_ns_q32", "attention_pair_ns_q32", "vision_item_ns",
                           "vision_patch_ns_q32"},
                          {}, context);
    ContextPrefillCost result{
        .chunk_ns = require_u64(value.at("chunk_ns"), std::string(context) + ".chunk_ns"),
        .token_ns_q32 =
            require_u64(value.at("token_ns_q32"), std::string(context) + ".token_ns_q32"),
        .attention_pair_ns_q32 = require_u64(value.at("attention_pair_ns_q32"),
                                             std::string(context) + ".attention_pair_ns_q32"),
        .vision_item_ns =
            require_u64(value.at("vision_item_ns"), std::string(context) + ".vision_item_ns"),
        .vision_patch_ns_q32 = require_u64(value.at("vision_patch_ns_q32"),
                                           std::string(context) + ".vision_patch_ns_q32"),
    };
    validate_prefill(result, context);
    return result;
}

Json transfer_direction_json(const ContextTransferCost& value) {
    return Json{{"batch_ns", value.batch_ns},
                {"operation_ns", value.operation_ns},
                {"ns_per_byte_q32", value.ns_per_byte_q32}};
}

Json transfer_json(const std::array<ContextTransferCost, 3>& value) {
    return Json{{"d2h", transfer_direction_json(value[0])},
                {"h2d", transfer_direction_json(value[1])},
                {"d2d", transfer_direction_json(value[2])}};
}

Json prefill_json(const ContextPrefillCost& value) {
    return Json{{"chunk_ns", value.chunk_ns},
                {"token_ns_q32", value.token_ns_q32},
                {"attention_pair_ns_q32", value.attention_pair_ns_q32},
                {"vision_item_ns", value.vision_item_ns},
                {"vision_patch_ns_q32", value.vision_patch_ns_q32}};
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open context-cost preset file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

Json parse_document(std::string_view text, std::string_view source_name) {
    try {
        return Json::parse(text.begin(), text.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument("invalid context-cost preset JSON in " +
                                    std::string(source_name) + ": " + error.what());
    }
}

const ContextCostMachinePreset* find_machine(const std::vector<ContextCostMachinePreset>& presets,
                                             std::string_view hardware_class) noexcept {
    const auto found = std::find_if(presets.begin(), presets.end(), [&](const auto& preset) {
        return preset.hardware_class == hardware_class;
    });
    return found == presets.end() ? nullptr : &*found;
}

const ContextPrefillPreset* find_prefill(const ContextCostMachinePreset& machine,
                                         std::string_view model_id,
                                         std::string_view weights_id) noexcept {
    const auto found =
        std::find_if(machine.prefill.begin(), machine.prefill.end(), [&](const auto& preset) {
            return preset.model_id == model_id && preset.weights_id == weights_id;
        });
    return found == machine.prefill.end() ? nullptr : &*found;
}

void validate_presets(const std::vector<ContextCostMachinePreset>& presets,
                      std::string_view context) {
    for (std::size_t machine_index = 0; machine_index < presets.size(); ++machine_index) {
        const ContextCostMachinePreset& machine = presets[machine_index];
        const std::string machine_context =
            std::string(context) + ".machines[" + std::to_string(machine_index) + "]";
        if (machine.hardware_class.empty()) {
            throw std::invalid_argument(machine_context + ".hardware_class is empty");
        }
        if (machine.transfer) {
            validate_transfer(*machine.transfer, machine_context + ".transfer");
        }
        if (!machine.transfer && machine.prefill.empty()) {
            throw std::invalid_argument(machine_context + " contains no cost data");
        }
        for (std::size_t prior = 0; prior < machine_index; ++prior) {
            if (presets[prior].hardware_class == machine.hardware_class) {
                throw std::invalid_argument("duplicate context-cost hardware_class: " +
                                            machine.hardware_class);
            }
        }
        for (std::size_t prefill_index = 0; prefill_index < machine.prefill.size();
             ++prefill_index) {
            const ContextPrefillPreset& prefill = machine.prefill[prefill_index];
            if (prefill.model_id.empty() || prefill.weights_id.empty()) {
                throw std::invalid_argument(machine_context + ".prefill has an empty identity");
            }
            validate_prefill(prefill.cost, machine_context + ".prefill[" +
                                               std::to_string(prefill_index) + "].coefficients");
            for (std::size_t prior = 0; prior < prefill_index; ++prior) {
                if (machine.prefill[prior].model_id == prefill.model_id &&
                    machine.prefill[prior].weights_id == prefill.weights_id) {
                    throw std::invalid_argument(
                        "duplicate context-cost prefill identity: " + machine.hardware_class + "/" +
                        prefill.model_id + "/" + prefill.weights_id);
                }
            }
        }
    }
}

Json parse_provenance(std::string_view provenance_json) {
    Json provenance;
    try {
        provenance = Json::parse(provenance_json.begin(), provenance_json.end());
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("invalid context-cost provenance JSON: ") +
                                    error.what());
    }
    if (!provenance.is_object()) {
        throw std::invalid_argument("context-cost provenance must be an object");
    }
    return provenance;
}

Json load_or_create_document(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return Json{{"schema_version", 2},
                    {"artifact_type", "ninfer_context_cost_presets"},
                    {"machines", Json::array()}};
    }
    const std::string text = read_text_file(path);
    (void)parse_context_cost_presets(text, path.string());
    return parse_document(text, path.string());
}

Json& find_or_append_machine(Json& document, std::string_view hardware_class) {
    for (Json& machine : document.at("machines")) {
        if (machine.at("hardware_class").get_ref<const std::string&>() == hardware_class) {
            return machine;
        }
    }
    document.at("machines")
        .push_back(Json{
            {"hardware_class", hardware_class}, {"transfer", nullptr}, {"prefill", Json::array()}});
    return document.at("machines").back();
}

void write_document_atomic(const std::filesystem::path& path, const Json& document) {
    const std::string serialized = document.dump(2) + '\n';
    (void)parse_context_cost_presets(serialized, "generated context-cost presets");
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }

    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("failed to open temporary context-cost preset: " +
                                         temporary.string());
            }
            output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
            if (!output) {
                throw std::runtime_error("failed to write temporary context-cost preset: " +
                                         temporary.string());
            }
        }
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace

std::uint64_t ContextCostModel::transfer_ns(ContextTransferDirection direction,
                                            TransferWork work) const noexcept {
    if (work.payload_bytes == 0 && work.copy_operations == 0) { return 0; }
    const std::size_t index = direction_index(direction);
    if (index >= transfer.size()) { return std::numeric_limits<std::uint64_t>::max(); }
    const ContextTransferCost& cost = transfer[index];
    const std::uint64_t operation_limited =
        saturating_add(cost.batch_ns, saturating_product(cost.operation_ns, work.copy_operations));
    const std::uint64_t bandwidth_limited =
        q32_product_ns(cost.ns_per_byte_q32, work.payload_bytes);
    return std::max(operation_limited, bandwidth_limited);
}

std::uint64_t ContextCostModel::prefill_ns(PrefillWork work) const noexcept {
    std::uint64_t result = saturating_product(prefill.chunk_ns, work.chunks);
    result = saturating_add(result, q32_product_ns(prefill.token_ns_q32, work.tokens));
    result =
        saturating_add(result, q32_product_ns(prefill.attention_pair_ns_q32, work.attention_pairs));
    result = saturating_add(result, saturating_product(prefill.vision_item_ns, work.vision_items));
    result =
        saturating_add(result, q32_product_ns(prefill.vision_patch_ns_q32, work.vision_patches));
    return result;
}

std::string context_cost_hardware_class(std::string_view gpu_name, int major, int minor) {
    std::string slug;
    slug.reserve(gpu_name.size() + 20);
    for (const unsigned char character : gpu_name) {
        if (std::isalnum(character)) {
            slug.push_back(static_cast<char>(std::tolower(character)));
        } else if (!slug.empty() && slug.back() != '-') {
            slug.push_back('-');
        }
    }
    while (!slug.empty() && slug.back() == '-') { slug.pop_back(); }
    if (slug.empty()) { slug = "gpu"; }
    if (slug != "nvidia" && !slug.starts_with("nvidia-")) { slug.insert(0, "nvidia-"); }
    return slug + "-sm" + std::to_string(major) + std::to_string(minor);
}

ContextCostModel generic_context_cost_model() {
    return ContextCostModel{.transfer = generic_context_transfer_cost(),
                            .prefill  = generic_context_prefill_cost()};
}

std::vector<ContextCostMachinePreset> parse_context_cost_presets(std::string_view json,
                                                                 std::string_view source_name) {
    const Json document = parse_document(json, source_name);
    require_exact_members(document, {"schema_version", "artifact_type", "machines"}, {},
                          source_name);
    if (require_u64(document.at("schema_version"), std::string(source_name) + ".schema_version") !=
        2) {
        throw std::invalid_argument("unsupported context-cost preset schema_version in " +
                                    std::string(source_name));
    }
    if (require_nonempty_string(document.at("artifact_type"),
                                std::string(source_name) + ".artifact_type") !=
        "ninfer_context_cost_presets") {
        throw std::invalid_argument("invalid context-cost artifact_type in " +
                                    std::string(source_name));
    }
    const Json& machines = document.at("machines");
    if (!machines.is_array()) {
        throw std::invalid_argument(std::string(source_name) + ".machines must be an array");
    }

    std::vector<ContextCostMachinePreset> result;
    result.reserve(machines.size());
    for (std::size_t machine_index = 0; machine_index < machines.size(); ++machine_index) {
        const Json& machine = machines[machine_index];
        const std::string context =
            std::string(source_name) + ".machines[" + std::to_string(machine_index) + "]";
        require_exact_members(machine, {"hardware_class", "transfer", "prefill"},
                              {"transfer_provenance"}, context);
        if (machine.contains("transfer_provenance") &&
            !machine.at("transfer_provenance").is_object()) {
            throw std::invalid_argument(context + ".transfer_provenance must be an object");
        }
        ContextCostMachinePreset parsed{
            .hardware_class =
                require_nonempty_string(machine.at("hardware_class"), context + ".hardware_class"),
        };
        if (!machine.at("transfer").is_null()) {
            parsed.transfer = parse_transfer(machine.at("transfer"), context + ".transfer");
        }
        const Json& prefill = machine.at("prefill");
        if (!prefill.is_array()) {
            throw std::invalid_argument(context + ".prefill must be an array");
        }
        parsed.prefill.reserve(prefill.size());
        for (std::size_t prefill_index = 0; prefill_index < prefill.size(); ++prefill_index) {
            const Json& value = prefill[prefill_index];
            const std::string prefill_context =
                context + ".prefill[" + std::to_string(prefill_index) + "]";
            require_exact_members(value, {"model_id", "weights_id", "coefficients"}, {"provenance"},
                                  prefill_context);
            if (value.contains("provenance") && !value.at("provenance").is_object()) {
                throw std::invalid_argument(prefill_context + ".provenance must be an object");
            }
            parsed.prefill.push_back(ContextPrefillPreset{
                .model_id =
                    require_nonempty_string(value.at("model_id"), prefill_context + ".model_id"),
                .weights_id = require_nonempty_string(value.at("weights_id"),
                                                      prefill_context + ".weights_id"),
                .cost = parse_prefill(value.at("coefficients"), prefill_context + ".coefficients"),
            });
        }
        result.push_back(std::move(parsed));
    }
    validate_presets(result, source_name);
    return result;
}

ResolvedContextCost resolve_context_cost(const ContextCostIdentity& identity,
                                         const std::filesystem::path& external_preset_path) {
    if (identity.hardware_class.empty() || identity.model_id.empty() ||
        identity.weights_id.empty()) {
        throw std::invalid_argument("context-cost identity contains an empty component");
    }

    ContextCostModel model{.transfer = generic_context_transfer_cost(),
                           .prefill  = generic_context_prefill_cost()};
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;

    const auto& compiled = compiled_context_cost_defaults();
    validate_presets(compiled, "compiled context-cost defaults");
    if (const ContextCostMachinePreset* machine = find_machine(compiled, identity.hardware_class)) {
        if (machine->transfer) {
            model.transfer  = *machine->transfer;
            transfer_source = ContextCostPresetSource::CompiledDefault;
        }
        if (const ContextPrefillPreset* prefill =
                find_prefill(*machine, identity.model_id, identity.weights_id)) {
            model.prefill  = prefill->cost;
            prefill_source = ContextCostPresetSource::CompiledDefault;
        }
    }

    if (!external_preset_path.empty()) {
        const auto external = parse_context_cost_presets(read_text_file(external_preset_path),
                                                         external_preset_path.string());
        if (const ContextCostMachinePreset* machine =
                find_machine(external, identity.hardware_class)) {
            if (machine->transfer) {
                model.transfer  = *machine->transfer;
                transfer_source = ContextCostPresetSource::External;
            }
            if (const ContextPrefillPreset* prefill =
                    find_prefill(*machine, identity.model_id, identity.weights_id)) {
                model.prefill  = prefill->cost;
                prefill_source = ContextCostPresetSource::External;
            }
        }
    }

    return ResolvedContextCost{
        .model = model,
        .summary =
            {
                .transfer_source = transfer_source,
                .prefill_source  = prefill_source,
                .hardware_class  = identity.hardware_class,
                .model_id        = identity.model_id,
                .weights_id      = identity.weights_id,
                .preset_path     = external_preset_path,
            },
    };
}

void upsert_context_transfer_cost_atomic(const std::filesystem::path& path,
                                         std::string_view hardware_class,
                                         const std::array<ContextTransferCost, 3>& transfer,
                                         std::string_view provenance_json) {
    if (path.empty()) { throw std::invalid_argument("context-cost preset output is empty"); }
    if (hardware_class.empty()) {
        throw std::invalid_argument("context-cost hardware_class is empty");
    }
    validate_transfer(transfer, "context-cost transfer");
    Json document                  = load_or_create_document(path);
    Json& machine                  = find_or_append_machine(document, hardware_class);
    machine["transfer"]            = transfer_json(transfer);
    machine["transfer_provenance"] = parse_provenance(provenance_json);
    write_document_atomic(path, document);
}

void upsert_context_prefill_cost_atomic(const std::filesystem::path& path,
                                        const ContextCostIdentity& identity,
                                        const ContextPrefillCost& prefill,
                                        std::string_view provenance_json) {
    if (path.empty()) { throw std::invalid_argument("context-cost preset output is empty"); }
    if (identity.hardware_class.empty() || identity.model_id.empty() ||
        identity.weights_id.empty()) {
        throw std::invalid_argument("context-cost identity contains an empty component");
    }
    validate_prefill(prefill, "context-cost prefill");
    Json document = load_or_create_document(path);
    Json& machine = find_or_append_machine(document, identity.hardware_class);
    Json replacement{{"model_id", identity.model_id},
                     {"weights_id", identity.weights_id},
                     {"coefficients", prefill_json(prefill)},
                     {"provenance", parse_provenance(provenance_json)}};
    Json& entries    = machine.at("prefill");
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const Json& value) {
        return value.at("model_id").get_ref<const std::string&>() == identity.model_id &&
               value.at("weights_id").get_ref<const std::string&>() == identity.weights_id;
    });
    if (found == entries.end()) {
        entries.push_back(std::move(replacement));
    } else {
        *found = std::move(replacement);
    }
    write_document_atomic(path, document);
}

} // namespace ninfer::runtime
