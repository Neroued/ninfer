#include "serve/http_server.h"

#include "product/statistics.h"
#include "serve/http_transport.h"

namespace ninfer::serve {
namespace {
using Json = nlohmann::json;

Json identity(const ServiceObservations& observations) {
    return {{"server_instance_id", observations.instance_id()},
            {"start_time_seconds", observations.start_seconds()},
            {"timestamp_unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count()}};
}

Json coordinate(const ParallelCoordinate& value) {
    return {{"rank", value.rank}, {"size", value.size}};
}
} // namespace

void HttpServer::register_monitor_routes() {
    server_.Get("/monitor/info", [this](const httplib::Request&, httplib::Response& res) {
        auto out                = identity(observations_);
        const auto& description = service_->description();
        auto options            = options_;
        options.launch.engine   = service_->engine_options();
        // configuration_view owns normalized presentation and credential redaction.
        for (const auto& item : product::configuration_view(options).items)
            out["configuration"][item.key] =
                std::visit([](const auto& value) { return Json(value); }, item.value);
        out["model"] = {
            {"id", public_model_id_},
            {"architecture", description.architecture},
            {"name", description.model_name},
            {"max_context", description.max_context},
            {"max_concurrency", description.max_concurrency},
            {"prefill_chunk", description.prefill_chunk},
            {"kv_cache", kv_cache_profile_info(description.kv_cache).name},
            {"context_token_capacity_estimate", description.context_token_capacity_estimate}};
        out["ranks"] = Json::array();
        for (const auto& r : description.ranks)
            out["ranks"].push_back({{"rank", r.rank},
                                    {"node", r.node},
                                    {"device", r.device},
                                    {"hardware", r.hardware},
                                    {"shared_physical_memory", r.shared_physical_memory},
                                    {"tp", coordinate(r.tp)},
                                    {"ep", coordinate(r.ep)},
                                    {"dp", coordinate(r.dp)},
                                    {"pp", coordinate(r.pp)},
                                    {"adp", coordinate(r.adp)}});
        out["placements"] = Json::array();
        for (const auto& p : description.placements)
            out["placements"].push_back({{"id", p.id}, {"attention_ranks", p.attention_ranks}});
        out["resources"] = product::startup_resources_json(description, service_->startup_report(),
                                                           service_->snapshot());
        res.set_header("Cache-Control", "no-store");
        res.set_content(out.dump(), "application/json");
    });
    server_.Get("/monitor/snapshot", [this](const httplib::Request&, httplib::Response& res) {
        const auto snapshot =
            observations_.normalize(service_->snapshot({.include_requests = true}));
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
        auto out       = identity(observations_);
        out["engine"]  = product::observation_json(snapshot, now);
        out["service"] = service_observation_json(observations_.snapshot(true));
        res.set_header("Cache-Control", "no-store");
        res.set_content(out.dump(), "application/json");
    });
    server_.Get("/monitor/events", [this](const httplib::Request&, httplib::Response& res) {
        auto subscription = observations_.events.subscribe();
        if (!subscription) {
            res.status = 503;
            res.set_content(Json{{"error", "monitor subscriber capacity unavailable"}}.dump(),
                            "application/json");
            return;
        }
        const auto hello = "event: connected\ndata: " + identity(observations_).dump() + "\n\n";
        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [subscription, hello](std::size_t offset, httplib::DataSink& sink) {
                std::atomic<bool> cancelled{false};
                SseTransport transport(sink, cancelled);
                try {
                    if (!offset) transport.write(hello);
                    while (!subscription->closed()) {
                        if (auto event = subscription->next(std::chrono::milliseconds(500)))
                            transport.write(*event);
                        if (transport.poll()) break;
                    }
                } catch (const ClientDisconnected&) {}
                subscription->close();
                return false;
            },
            [subscription](bool) { subscription->close(); });
    });
}
} // namespace ninfer::serve
