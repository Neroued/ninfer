#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/cli.h"
#include "qus/text/text_runner.h"
#include "qus/text/tokenizer.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    try {
        const qus::text::CliOptions cli = qus::text::parse_cli(argc, argv);
        if (cli.help_requested) {
            std::cout << qus::text::usage_text(argv[0]);
            return 0;
        }

        qus::text::QwenTokenizer tokenizer(cli.tokenizer_path);
        const std::vector<int> stop_token_ids =
            qus::text::resolve_stop_token_ids(tokenizer, cli.stop_token_ids);

        qus::EngineOptions engine_options;
        engine_options.device         = cli.device;
        engine_options.max_ctx        = cli.max_context;
        engine_options.stop_token_ids = stop_token_ids;
        qus::Engine engine(engine_options);
        engine.load(cli.weights_path);

        const std::vector<qus::text::ChatMessage> messages =
            cli.messages_path.empty() ? qus::text::messages_from_prompt(cli.prompt)
                                      : qus::text::read_messages_json(cli.messages_path);

        qus::text::TextGenerationOptions generation_options;
        generation_options.max_new_tokens = cli.max_new;
        generation_options.raw_output     = cli.output_mode == qus::text::OutputMode::Raw;
        generation_options.stop_token_ids = stop_token_ids;

        qus::text::TextGenerationRunner runner(tokenizer, engine);

        const auto start                       = std::chrono::steady_clock::now();
        const qus::text::TextGenerationResult result = runner.generate(messages, generation_options);
        const auto end                         = std::chrono::steady_clock::now();

        std::cout << result.text;
        if (result.text.empty() || result.text.back() != '\n') { std::cout << '\n'; }

        if (cli.print_token_ids) {
            std::cerr << "generated_ids=";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double tok_s =
            seconds > 0.0 ? static_cast<double>(result.generated_token_ids.size()) / seconds : 0.0;
        std::cerr << "generated=" << result.generated_token_ids.size() << " elapsed_s=" << seconds
                  << " tok_s=" << tok_s << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << qus::text::usage_text(argv[0]);
        return 1;
    }

    return 0;
}
