/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

#include "compiled_grammar_impl.h"
#include "support/int_set.h"
#include "support/json_parse.h"
#include "support/json_serializer.h"
#include "testing.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/exception.h"

namespace xgrammar {

std::optional<FSMWithStartEnd> NormalizeRepetitionUnit(const FSMWithStartEnd& fsm) {
  constexpr int kMaxUnitStates = 256;
  if (fsm.NumStates() > kMaxUnitStates) return std::nullopt;
  std::unordered_set<int> reachable;
  fsm.GetReachableStates(&reachable);
  for (int state : reachable) {
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      if (!edge.IsCharRange() && !edge.IsEpsilon()) return std::nullopt;
    }
  }
  auto result = fsm.ToDFA(kMaxUnitStates);
  if (result.IsErr()) return std::nullopt;
  auto unit = std::move(result).Unwrap();
  if (!RepetitionUnitFirstBytes(unit.ToCompact())) return std::nullopt;
  return unit;
}

std::optional<std::bitset<256>> RepetitionUnitFirstBytes(const CompactFSMWithStartEnd& fsm) {
  if (fsm.IsEndState(fsm.GetStart())) return std::nullopt;
  std::unordered_set<int> reachable;
  fsm.GetReachableStates(&reachable);
  std::unordered_map<int, std::vector<int>> predecessors;
  std::bitset<256> first;
  for (int state : reachable) {
    std::bitset<256> seen;
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      if (!edge.IsCharRange() || edge.max > 255 || fsm.IsEndState(state)) return std::nullopt;
      for (int byte = edge.min; byte <= edge.max; ++byte) {
        if (seen[byte]) return std::nullopt;
        seen.set(byte);
      }
      predecessors[edge.target].push_back(state);
    }
    if (state == fsm.GetStart()) first = seen;
  }
  std::unordered_set<int> productive(fsm.GetEnds().begin(), fsm.GetEnds().end());
  std::vector<int> pending(fsm.GetEnds().begin(), fsm.GetEnds().end());
  for (size_t i = 0; i < pending.size(); ++i) {
    for (int state : predecessors[pending[i]]) {
      if (productive.insert(state).second) pending.push_back(state);
    }
  }
  for (int state : reachable) {
    if (!productive.count(state)) return std::nullopt;
  }
  return first;
}

namespace {

using ByteSet = std::bitset<256>;

/*!
 * \brief The next state after `byte` in an FSM whose outgoing character ranges do not overlap.
 * \return -1 when no edge accepts the byte.
 */
int NextStateOnByte(const CompactFSM& fsm, int state, uint8_t byte) {
  for (const auto& edge : fsm.GetEdges(state)) {
    if (edge.min <= byte && byte <= edge.max) {
      return edge.target;
    }
  }
  return -1;
}

// Unit rules have no annotations or rule calls whose semantics a byte DFA could erase.
bool IsRepetitionUnitRule(const Grammar& grammar, int32_t rule_id, ByteSet* first_bytes) {
  const auto& rule = grammar->GetRule(rule_id);
  if (rule_id == grammar->GetRootRuleId() || rule.lookahead_assertion_id != -1 ||
      rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() || rule.is_lazy ||
      rule.temperature.has_value() || grammar->GetSuffixStopInfo(rule_id) != nullptr) {
    return false;
  }
  const auto& rule_fsm = grammar->per_rule_fsms[rule_id];
  if (!rule_fsm) return false;
  const auto first = RepetitionUnitFirstBytes(rule_fsm->GetFsm());
  if (!first) return false;
  *first_bytes = *first;
  return true;
}

// Exit bytes accepted by all/any references to the shared body.
struct RepetitionFollowBytes {
  ByteSet under_all;
  ByteSet under_any;
};

std::vector<std::optional<RepetitionFollowBytes>> FindCountedRepetitionBodies(const Grammar& grammar
) {
  const int32_t num_rules = grammar->NumRules();
  std::vector<std::optional<ByteSet>> first_bytes(num_rules);
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    // The matcher bypasses the fast path when character budgets are enforced.
    if (grammar->GetRule(rule_id).max_chars >= 0) {
      return std::vector<std::optional<RepetitionFollowBytes>>(num_rules);
    }
    ByteSet bytes;
    if (IsRepetitionUnitRule(grammar, rule_id, &bytes)) {
      first_bytes[rule_id] = bytes;
    }
  }

  std::vector<bool> disqualified(num_rules, false);
  std::vector<bool> referenced(num_rules, false);
  std::vector<RepetitionFollowBytes> follow_bytes(num_rules, {ByteSet().set(), ByteSet()});
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    if (const auto* suffix_stop_info = grammar->GetSuffixStopInfo(rule_id)) {
      for (int32_t ref : {suffix_stop_info->body_rule_id, suffix_stop_info->marker_rule_id}) {
        if (ref >= 0) {
          disqualified[ref] = true;
        }
      }
    }
    const auto& rule_fsm = grammar->per_rule_fsms[rule_id];
    if (!rule_fsm.has_value()) {
      continue;
    }
    std::unordered_set<int> reachable_states;
    rule_fsm->GetFsm().GetReachableStates(&reachable_states);
    for (int state : reachable_states) {
      for (const auto& edge : rule_fsm->GetFsm().GetFsm().GetEdges(state)) {
        if (edge.IsRuleRef()) {
          disqualified[edge.GetRefRuleId()] = true;
          continue;
        }
        if (!edge.IsRepeatRef()) {
          continue;
        }
        const auto info = grammar->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex());
        const int32_t body_rule_id = info.RuleId();
        if (!first_bytes[body_rule_id].has_value()) {
          continue;
        }
        referenced[body_rule_id] = true;
        if (rule_fsm->GetFsm().IsEndState(edge.target)) {
          disqualified[body_rule_id] = true;
          continue;
        }
        ByteSet bytes;
        for (const auto& follow_edge : rule_fsm->GetFsm().GetFsm().GetEdges(edge.target)) {
          if (!follow_edge.IsCharRange() || follow_edge.max > 0xFF) {
            disqualified[body_rule_id] = true;
            break;
          }
          for (int byte = follow_edge.min; byte <= follow_edge.max; ++byte) {
            bytes.set(byte);
          }
        }
        if ((bytes & *first_bytes[body_rule_id]).any()) {
          disqualified[body_rule_id] = true;
        }
        follow_bytes[body_rule_id].under_all &= bytes;
        follow_bytes[body_rule_id].under_any |= bytes;
      }
    }
  }

  std::vector<std::optional<RepetitionFollowBytes>> result(num_rules);
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    if (first_bytes[rule_id].has_value() && referenced[rule_id] && !disqualified[rule_id]) {
      result[rule_id] = follow_bytes[rule_id];
    }
  }
  return result;
}

/*! \brief How the body state of a counted repetition decides a token. */
struct RepetitionTokenClass {
  enum Kind {
    // Legal exactly when the budget left covers `num_repetitions`.
    kBudget,
    // Leaves the repetition on its last byte: legal exactly when the budget left covers
    // `num_repetitions` and they complete the lower bound.
    kExit,
    // Never legal from this state.
    kRejected,
    // Depends on what follows the repetition: left to the matcher's replay.
    kUndecided,
  };
  Kind kind;
  int32_t num_repetitions = 0;
};

// Classify from any unit state. Prefix-free completion resets the DFA to its start;
// an incomplete final unit consumes one more repetition. Tokens crossing beyond an exit
// byte retain the ordinary parser replay so the parent's full continuation is respected.
RepetitionTokenClass ClassifyRepetitionToken(
    const CompactFSMWithStartEnd& fsm,
    const RepetitionFollowBytes& follow_bytes,
    const std::string& token,
    int initial_state
) {
  int32_t num_repetitions = 0;
  int state = initial_state;
  for (size_t i = 0; i < token.size(); ++i) {
    const auto byte = static_cast<uint8_t>(token[i]);
    const int next_state = NextStateOnByte(fsm.GetFsm(), state, byte);
    if (next_state == -1) {
      if (state != fsm.GetStart() || num_repetitions == 0 || !follow_bytes.under_any[byte]) {
        return {RepetitionTokenClass::kRejected};
      }
      if (i + 1 == token.size() && follow_bytes.under_all[byte]) {
        // The parent consumes this last byte right after `num_repetitions` units.
        return {RepetitionTokenClass::kExit, num_repetitions};
      }
      return {RepetitionTokenClass::kUndecided};
    }
    state = next_state;
    if (fsm.IsEndState(state)) {
      ++num_repetitions;
      state = fsm.GetStart();
    }
  }
  // A token ending inside a unit needs one more repetition to complete it.
  if (state != fsm.GetStart()) {
    ++num_repetitions;
  }
  return {RepetitionTokenClass::kBudget, num_repetitions};
}

}  // namespace

void PopulateRepeatMasks(
    const Grammar& grammar,
    const TokenizerInfo& tokenizer_info,
    std::unordered_map<ParserState, AdaptiveTokenMask, StateHashForCache, StateEqualForCache>* cache
) {
  using StoreType = AdaptiveTokenMask::StoreType;
  const auto& sorted_vocab = tokenizer_info.GetSortedDecodedVocab();
  const auto counted_repetition_bodies = FindCountedRepetitionBodies(grammar);
  for (auto& [state, mask] : *cache) {
    mask.repeat_interior_counts.clear();
    mask.repeat_interior_bitsets.clear();
    mask.repeat_interior_tokens.clear();
    mask.repeat_exit_tokens.clear();
    if (state.rule_id < 0 || !counted_repetition_bodies[state.rule_id].has_value()) {
      continue;
    }
    const auto& fsm = grammar->per_rule_fsms[state.rule_id]->GetFsm();
    // Bucket the budget-decided tokens by the repetitions they consume, keeping the
    // sorted-vocabulary order.
    std::map<int32_t, std::vector<int32_t>> tokens_by_repetitions;
    std::vector<int32_t> decided_indices;
    ByteSet first_bytes;
    for (const auto& edge : fsm.GetFsm().GetEdges(state.element_id)) {
      for (int byte = edge.min; byte <= edge.max; ++byte) first_bytes.set(byte);
    }
    // Tokens outside these first-byte intervals are already statically rejected. This also
    // works after deserialization, when the budget-dependent tokens are absent from the mask.
    auto first_token = [&](int byte) {
      return std::lower_bound(
          sorted_vocab.begin(), sorted_vocab.end(), byte, [](const auto& token, int value) {
            return token.second.empty() || static_cast<uint8_t>(token.second.front()) < value;
          });
    };
    for (int byte = 0; byte < 256;) {
      if (!first_bytes[byte]) {
        ++byte;
        continue;
      }
      const auto begin = first_token(byte);
      do {
        ++byte;
      } while (byte < 256 && first_bytes[byte]);
      const auto end = first_token(byte);
      for (auto token = begin; token != end; ++token) {
        const int32_t index = static_cast<int32_t>(token - sorted_vocab.begin());
        const auto token_class =
            ClassifyRepetitionToken(fsm, *counted_repetition_bodies[state.rule_id],
                                    sorted_vocab[index].second, state.element_id);
        if (token_class.kind == RepetitionTokenClass::kBudget) {
          tokens_by_repetitions[token_class.num_repetitions].push_back(index);
        } else if (token_class.kind == RepetitionTokenClass::kExit) {
          mask.repeat_exit_tokens.emplace_back(token_class.num_repetitions,
                                              sorted_vocab[index].first);
          decided_indices.push_back(index);
        } else if (token_class.kind == RepetitionTokenClass::kRejected) {
          decided_indices.push_back(index);
        }
      }
    }
    if (tokens_by_repetitions.empty() && mask.repeat_exit_tokens.empty()) {
      continue;
    }

    size_t interior_size = 0;
    for (const auto& [count, indices] : tokens_by_repetitions) interior_size += indices.size();
    const bool sparse = interior_size < AdaptiveTokenMask::USE_BITSET_THRESHOLD;
    for (auto& [num_repetitions, indices] : tokens_by_repetitions) {
      DynamicBitset cumulative;
      if (!sparse) {
        cumulative = mask.repeat_interior_bitsets.empty()
                         ? DynamicBitset(tokenizer_info.GetVocabSize())
                         : mask.repeat_interior_bitsets.back();
      }
      for (int32_t index : indices) {
        if (sparse)
          mask.repeat_interior_tokens.emplace_back(num_repetitions, sorted_vocab[index].first);
        else
          cumulative.Set(sorted_vocab[index].first, true);
        decided_indices.push_back(index);
      }
      if (!sparse) {
        mask.repeat_interior_counts.push_back(num_repetitions);
        mask.repeat_interior_bitsets.push_back(std::move(cumulative));
      }
    }
    std::sort(decided_indices.begin(), decided_indices.end());

    // Take the decided tokens out of every static class: the matcher accepts the budget and exit
    // tokens that fit the repetition bounds, and nothing else of them. A kRejected mask would
    // accept them implicitly, so it is turned into an accepted bitset first; the mask then never
    // accepts a token it does not list, whatever the other states of the row decide.
    if (mask.store_type == StoreType::kRejected) {
      std::vector<int32_t> not_accepted = mask.rejected_indices;
      IntsetUnion(&not_accepted, mask.uncertain_indices);
      IntsetUnion(&not_accepted, decided_indices);
      mask.accepted_bitset = DynamicBitset(tokenizer_info.GetVocabSize());
      auto it = not_accepted.begin();
      for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
        while (it != not_accepted.end() && *it < index) {
          ++it;
        }
        if (it == not_accepted.end() || *it != index) {
          mask.accepted_bitset.Set(sorted_vocab[index].first, true);
        }
      }
      mask.rejected_indices.clear();
      mask.store_type = StoreType::kAcceptedBitset;
    } else if (mask.store_type == StoreType::kAcceptedBitset) {
      for (int32_t index : decided_indices) {
        mask.accepted_bitset.Set(sorted_vocab[index].first, false);
      }
    } else {
      IntsetDifference(&mask.accepted_indices, decided_indices);
    }
    IntsetDifference(&mask.uncertain_indices, decided_indices);
  }
}

/******************* AdaptiveTokenMask *******************/

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD
                   ? StoreType::kAcceptedBitset
               : size_acc < size_rej ? StoreType::kAccepted
                                     : StoreType::kRejected;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else if (store_type == StoreType::kAccepted) {
    this->accepted_indices = accepted_indices;
  } else {
    this->rejected_indices = rejected_indices;
  }

  this->uncertain_indices = uncertain_indices;
}

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD ? StoreType::kAcceptedBitset : StoreType::kAccepted;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kAccepted);
    this->accepted_indices = accepted_indices;
  }
  this->uncertain_indices = uncertain_indices;
}

std::string AdaptiveTokenMask::Print(const TokenizerInfo& tokenizer_info) const {
  constexpr int kMaxPrintTokens = 100;
  std::stringstream ss;
  const auto& sorted_decoded_vocab = tokenizer_info.GetSortedDecodedVocab();
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::unordered_set<int32_t> uncertain_indices_set(
      uncertain_indices.begin(), uncertain_indices.end()
  );

  accepted_indices.reserve(sorted_decoded_vocab.size());
  rejected_indices.reserve(sorted_decoded_vocab.size());

  if (store_type == StoreType::kAcceptedBitset) {
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      if (accepted_bitset[sorted_decoded_vocab[i].first]) {
        accepted_indices.push_back(i);
      } else {
        rejected_indices.push_back(i);
      }
    }
  } else if (store_type == StoreType::kAccepted) {
    accepted_indices = this->accepted_indices;
    // Reject indices = [0, sorted_decoded_vocab.size()) \ accepted_indices \ uncertain_indices
    int acc_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] < i) {
        ++acc_ptr;
      }
      if (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      rejected_indices.push_back(i);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kRejected);
    rejected_indices = this->rejected_indices;
    // Accepted indices = [0, sorted_decoded_vocab.size()) \ rejected_indices \ uncertain_indices
    int rej_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] < i) {
        ++rej_ptr;
      }
      if (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      accepted_indices.push_back(i);
    }
  }

  std::string storage_type_str = store_type == StoreType::kAcceptedBitset ? "AcceptedBitset"
                                 : store_type == StoreType::kAccepted     ? "Accepted"
                                                                          : "Rejected";

  ss << "AdaptiveTokenMask(num_tokens=" << sorted_decoded_vocab.size()
     << ", accepted_num=" << accepted_indices.size() << ", rejected_num=" << rejected_indices.size()
     << ", uncertain_num=" << uncertain_indices.size() << ", storage_type=" << storage_type_str
     << ",\n";

  // Convert indices to token ids for printing
  std::vector<int32_t> accepted_token_ids;
  std::vector<int32_t> rejected_token_ids;
  std::vector<int32_t> uncertain_token_ids;
  accepted_token_ids.reserve(accepted_indices.size());
  rejected_token_ids.reserve(rejected_indices.size());
  uncertain_token_ids.reserve(uncertain_indices.size());

  for (auto idx : accepted_indices) {
    accepted_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(accepted_token_ids.begin(), accepted_token_ids.end());
  for (auto idx : rejected_indices) {
    rejected_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(rejected_token_ids.begin(), rejected_token_ids.end());
  for (auto idx : uncertain_indices) {
    uncertain_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(uncertain_token_ids.begin(), uncertain_token_ids.end());

  ss << "accepted=" << PrintTokenByIds(accepted_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nrejected=" << PrintTokenByIds(rejected_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nuncertain=" << PrintTokenByIds(uncertain_token_ids, tokenizer_info, kMaxPrintTokens)
     << "\n)";
  return ss.str();
}

/************** CompiledGrammar::Impl **************/

picojson::value SerializeJSONValue(const CompiledGrammar::Impl& impl) {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(impl.grammar);
  result["tokenizer_metadata"] = impl.tokenizer_info->DumpMetadataValue();
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue(impl.adaptive_token_mask_cache);
  return picojson::value(result);
}

std::optional<SerializationError> DeserializeJSONValue(
    CompiledGrammar::Impl* impl,
    const picojson::value& json_value,
    const TokenizerInfo& tokenizer_info
) {
  const auto& type_name = "CompiledGrammar";
  if (!json_value.is<picojson::object>()) {
    return ConstructDeserializeError("Expect an object", type_name);
  }
  const auto& object = json_value.get<picojson::object>();
  if (object.find("grammar") == object.end()) {
    return ConstructDeserializeError("Expect a 'grammar' field", type_name);
  }
  if (auto error = AutoDeserializeJSONValue(&(impl->grammar), object["grammar"], type_name)) {
    return error;
  }
  if (impl->grammar.IsNull()) {
    return ConstructDeserializeError("Expect a non-null grammar", type_name);
  }
  if (object.find("tokenizer_metadata") == object.end()) {
    return ConstructDeserializeError("Expect a 'tokenizer_metadata' field", type_name);
  }
  const auto& tokenizer_metadata = object["tokenizer_metadata"];
  if (auto error = tokenizer_info->CheckMetadataMatch(tokenizer_metadata)) {
    return ConstructDeserializeError(
        std::string("Tokenizer metadata mismatch: ") + error->what(), type_name
    );
  }
  impl->tokenizer_info = tokenizer_info;
  if (object.find("adaptive_token_mask_cache") == object.end()) {
    return ConstructDeserializeError("Expect a 'adaptive_token_mask_cache' field", type_name);
  }
  if (auto error = AutoDeserializeJSONValue(
          &(impl->adaptive_token_mask_cache), object["adaptive_token_mask_cache"], type_name
      )) {
    return error;
  }
  // The masks index sorted_decoded_vocab and are OR-ed into a vocab_size-bit bitset, so their
  // contents must match the tokenizer they are deserialized with.
  const int64_t num_sorted_tokens = tokenizer_info.GetSortedDecodedVocab().size();
  auto indices_ok = [&](const std::vector<int32_t>& indices) {
    return std::all_of(indices.begin(), indices.end(), [&](int32_t index) {
      return index >= 0 && index < num_sorted_tokens;
    });
  };
  for (const auto& [state, mask] : impl->adaptive_token_mask_cache) {
    // PopulateRepeatMasks below looks the state's rule up in the grammar.
    if (state.rule_id >= impl->grammar->NumRules()) {
      return ConstructDeserializeError(
          "adaptive_token_mask_cache contains a state whose rule does not exist", type_name
      );
    }
    using StoreType = AdaptiveTokenMask::StoreType;
    const bool store_type_ok = mask.store_type == StoreType::kAccepted ||
                               mask.store_type == StoreType::kRejected ||
                               mask.store_type == StoreType::kAcceptedBitset;
    const bool bitset_ok = mask.store_type != StoreType::kAcceptedBitset ||
                           mask.accepted_bitset.Size() == tokenizer_info.GetVocabSize();
    if (!store_type_ok || !bitset_ok || !indices_ok(mask.accepted_indices) ||
        !indices_ok(mask.rejected_indices) || !indices_ok(mask.uncertain_indices)) {
      return ConstructDeserializeError(
          "adaptive_token_mask_cache contains a mask that does not match the tokenizer", type_name
      );
    }
  }
  // The repeat-interior fast path is derived from the grammar, so it is rebuilt here instead of
  // being part of the serialized form.
  PopulateRepeatMasks(impl->grammar, tokenizer_info, &impl->adaptive_token_mask_cache);
  return std::nullopt;
}

/************** CompiledGrammar **************/

std::size_t MemorySize(const CompiledGrammar::Impl& impl) {
  return MemorySize(impl.grammar) + MemorySize(impl.adaptive_token_mask_cache);
}

std::size_t CompiledGrammar::MemorySizeBytes() const { return MemorySize(*pimpl_); }

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

/*! \brief Return the serialized JSON string of the compiled grammar. */
std::string CompiledGrammar::SerializeJSON() const { return AutoSerializeJSON(*this, true); }

/*! \brief Deserialize a compiled grammar from a JSON string and tokenizer info. */
std::variant<CompiledGrammar, SerializationError> CompiledGrammar::DeserializeJSON(
    const std::string& json_string, const TokenizerInfo& tokenizer_info
) {
  picojson::value json_value;
  if (auto error = ParseJSON(json_value, json_string); !error.empty()) {
    return InvalidJSONError("Failed to parse JSON: " + error);
  }
  if (!json_value.is<picojson::object>()) {
    return DeserializeFormatError("Expect an object");
  }
  const auto& object = json_value.get<picojson::object>();
  if (auto error = SerializeVersion::Check(object)) {
    return error.value();
  }
  auto impl = std::make_shared<CompiledGrammar::Impl>();
  if (auto error = DeserializeJSONValue(impl.get(), json_value, tokenizer_info)) {
    return error.value();
  }
  return CompiledGrammar(std::move(impl));
}

}  // namespace xgrammar
