# Atropos Environments

This directory contains a collection of ready-to-use Reinforcement Learning environments for use with the Atropos framework. Each environment is a microservice designed to generate training data for a specific task.

## Available Environments

Below is an overview of some of the key environments available in this repository.

### `npc_arena_server.py` - NPC Arena
- **Purpose:** Fine-tune and evaluate language models to act as Non-Player Characters (NPCs) for games.
- **Method:** This environment uses a Reinforcement Learning with AI Feedback (RLAIF) approach. A "judge" model scores the trainee model's ability to roleplay as different characters based on predefined "character sheets."
- **Features:**
    - **Dynamic Personality Training:** Train a single model to adopt multiple personalities based on a system prompt.
    - **Quality-Based Scoring:** Uses an AI judge to evaluate dialogue on criteria like in-character consistency, lore accuracy, and engagingness.
    - **Extensible:** Includes a clear integration point for a Retrieval-Augmented Generation (RAG) system to allow NPCs to respond to dynamic, real-time game events.
- **Further Reading:** For a deep dive into the dozen different fine-tuning methods that informed the design of this environment, see the full research report: [`npc_finetuning_report.md`](../npc_finetuning_report.md).

### `rlaif_server.py` - Generic RLAIF
- **Purpose:** A generic environment for steering a model's behavior based on an abstract preference string, using an AI judge. This was the foundation for the `npc_arena_server.py`.

### `tool_calling_server.py` - Tool Calling
- **Purpose:** Train models to effectively use external tools and APIs. This is critical for creating agents that can interact with other systems.

### `gsm8k_server.py` - Math Problem Solving
- **Purpose:** Fine-tune models to solve grade-school math problems, a benchmark for evaluating reasoning capabilities.

... *(and other environments)* ...

## Contributing New Environments

We welcome contributions of new and interesting environments! If you would like to contribute, please see our main [CONTRIBUTING.md](../CONTRIBUTING.md) guide and place your new environment in the `environments/community/` subdirectory.