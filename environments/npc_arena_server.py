import asyncio
import copy
import random
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# === NPC Arena: Character Sheets ===
# This data structure defines the various NPC personalities the model can adopt.
# Each character has a system prompt that sets the scene and a personality description.
NPC_CHARACTER_SHEETS = [
    {
        "name": "Kaelen, the Grumpy Blacksmith",
        "system_prompt": "You are Kaelen, a 250-year-old dwarven blacksmith in the town of Stonehaven. You are a master of your craft but have grown weary of adventurers. You are suspicious of outsiders, especially elves. Your responses should be short, gruff, and to the point. You are not an AI assistant; you are Kaelen.",
        "personality": "Gruff, terse, proud, world-weary, slightly prejudiced against elves."
    },
    {
        "name": "Lyra, the Mischievous Sprite",
        "system_prompt": "You are Lyra, a tiny, ancient sprite who lives in the Whispering Woods. You have seen empires rise and fall, and it has given you a playful and enigmatic sense of humor. You speak only in riddles and never give a straight answer. You are not an AI assistant; you are Lyra.",
        "personality": "Mischievous, enigmatic, playful, speaks in riddles, ancient wisdom hidden beneath humor."
    },
    {
        "name": "Sir Reginald, the Valiant Knight",
        "system_prompt": "You are Sir Reginald, a young, idealistic knight sworn to protect the kingdom of Eldoria. You are earnest, polite, and always eager to help. You speak in a formal, chivalrous manner and address others with respect. You are a bastion of hope. You are not an AI assistant; you are Sir Reginald.",
        "personality": "Brave, idealistic, formal, polite, chivalrous, noble."
    },
]

# === NPC Arena: Judging Criteria ===
# This is the prompt used by the JUDGE model to score the trainee's responses.
# It instructs the judge to evaluate responses based on specific NPC criteria.

JUDGE_SYSTEM_PROMPT = """You are a meticulous AI Game Master and Dialogue Coach. Your task is to evaluate two potential responses from a trainee language model attempting to roleplay as a specific Non-Player Character (NPC).

You must analyze the responses based on the provided Character Sheet and select the one that is superior. Your evaluation should be based on the following criteria:
1.  **In-Character Consistency:** How well does the response align with the NPC's defined personality, tone, and backstory?
2.  **Lore Accuracy:** Does the response respect the established facts of the character and world?
3.  **Engagingness:** Is the response interesting, creative, and likely to encourage further interaction?

First, you will deeply consider the problem and deliberate with yourself via systematic reasoning processes to come to a correct solution. Enclose your thoughts and internal monologue inside <think> </think> tags.

After your reasoning, you must provide your final choice in the format: \\boxed{A or B}."""

NPC_JUDGE_PROMPT_FORMAT_STR = """[Character Sheet]
Name: {name}
Personality: {personality}

[Player's Line]
{conversation}

[Response A]
{response_a}

[Response B]
{response_b}

[END]

Based on the Character Sheet provided, please choose the response (A or B) that is a better example of roleplaying. Remember to follow the criteria: In-Character Consistency, Lore Accuracy, and Engagingness.

Think through your decision first, then provide your final answer in the format: \\boxed{{A or B}}."""


class NPCArenaEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.judgement_strings = list()
        self.player_prompts = list()
        self.iter = 0

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=2,
            use_wandb=True,
            max_num_workers=512 * 3 * 4,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=10000,
            max_token_length=8192,
            score_buffer_size=4,
            wandb_name="npc_arena",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if len(self.judgement_strings) > 0:
            # setup wandb table
            table = wandb.Table(columns=["resp_a", "resp_b", "sample_judgement"])
            for item in self.judgement_strings:
                table.add_data(item[0], item[1], item[2])
            self.judgement_strings.clear()
            wandb_metrics["train/judgement_table"] = table
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # A simple list of generic player questions to prompt the NPCs.
        self.player_prompts = [
            "What's the news around town?",
            "Have you seen anything unusual lately?",
            "Can you help me with something?",
            "I'm looking for the lost amulet of Eldoria.",
            "Tell me about yourself.",
        ]
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question, answer):
        pass

    async def evaluate(self, *args, **kwargs):
        pass

    async def get_dynamic_lore_from_rag(self, player_prompt: str) -> str:
        """
        **Placeholder for RAG Integration**

        This function simulates retrieving dynamic, real-time information from a
        knowledge base (e.g., a vector database of game lore, quest status, etc.).
        In a real implementation, this would involve:
        1.  Converting the `player_prompt` into an embedding.
        2.  Querying a vector database to find relevant documents.
        3.  Formatting the retrieved documents into a context string.

        This demonstrates the pathway to a Hybrid (RAG + PEFT/RLAIF) approach,
        as detailed in the research report `npc_finetuning_report.md`.
        By injecting this context, the NPC can respond to dynamic game events.
        """
        # Placeholder logic: If the player mentions a specific item, provide context.
        if "amulet of eldoria" in player_prompt.lower():
            return "RAG Context: The Amulet of Eldoria is rumored to be hidden in the Dragon's Maw cave, guarded by ancient spirits. The player has not yet started the 'Dragon's Amulet' quest."
        return ""  # Return empty string if no relevant context is found.

    async def collect_trajectories(self, item: Dict) -> Tuple[Optional[ScoredDataGroup], List]:
        character_sheet = item["character_sheet"]
        player_prompt = item["player_prompt"]

        # === RAG Integration Point (Method #12: Hybrid RAG + PEFT/RLAIF) ===
        # 1. Fetch dynamic context from our simulated RAG pipeline.
        rag_context = await self.get_dynamic_lore_from_rag(player_prompt)
        system_prompt = character_sheet["system_prompt"]
        if rag_context:
            # Augment the system prompt with real-time world knowledge.
            system_prompt += f"\n\n[World Knowledge]\n{rag_context}"
        # =================================================================

        # 2. Construct the chat history for the trainee model using Prompt Engineering (Method #2)
        chat = [
            {"role": "system", "content": system_prompt},  # Use the potentially augmented system prompt
            {"role": "user", "content": player_prompt},
        ]

        # 3. Check token length to avoid errors
        if len(self.tokenizer.apply_chat_template(chat)) >= (self.config.max_token_length) - (
            self.config.max_token_length // 2
        ):
            return None, []  # Skip if the prompt is too long

        # 4. Generate two different responses from the trainee model
        chat_completions = await self.server.chat_completion(
            messages=chat,
            n=2,
            max_tokens=self.config.max_token_length // 3,
        )

        # 5. Prepare the data for the judge model
        to_score = {
            "character_sheet": character_sheet,
            "player_prompt": player_prompt,
            "responses": [comp.message.content for comp in chat_completions.choices],
            "finish_reasons": [comp.finish_reason for comp in chat_completions.choices],
        }

        # 6. Call the score method to get the judged scores (Method #9: RLHF/RLAIF)
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    async def score(self, scoring_data: Dict) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        character_sheet = scoring_data["character_sheet"]
        player_prompt = scoring_data["player_prompt"]
        responses = scoring_data["responses"]
        finish_reasons = scoring_data["finish_reasons"]

        # Handle cases where one or both responses were too long
        if any([reason == "length" for reason in finish_reasons]):
            for i, response_content in enumerate(responses):
                chat_history = [
                    {"role": "system", "content": character_sheet["system_prompt"]},
                    {"role": "user", "content": player_prompt},
                    {"role": "assistant", "content": response_content},
                ]
                out_dict = tokenize_for_trainer(self.tokenizer, chat_history)
                scores["tokens"].append(out_dict["tokens"])
                scores["masks"].append(out_dict["masks"])
                # Penalize responses that were cut off
                scores["scores"].append(1.0 if finish_reasons[i] != "length" else -1.0)
            return scores

        # Format the prompt for the judge model
        judge_prompt = NPC_JUDGE_PROMPT_FORMAT_STR.format(
            name=character_sheet["name"],
            personality=character_sheet["personality"],
            conversation=player_prompt,
            response_a=responses[0],
            response_b=responses[1],
        )

        # Get judgement from the judge model (we can ask for multiple judgements to get a consensus)
        judgements = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            n=3,  # Request 3 judgements to reduce variance
            max_tokens=self.config.max_token_length,
        )

        # Save an example judgement to wandb for inspection
        if len(judgements.choices) > 0:
            self.judgement_strings.append(
                (responses[0], responses[1], judgements.choices[0].message.content)
            )

        # Calculate scores from judgements
        score_a = 0.0
        score_b = 0.0
        for choice in judgements.choices:
            chosen_val = choice.message.content.split("\\boxed{")[-1].strip().replace("}", "")
            if chosen_val == "A":
                score_a += 1.0
            elif chosen_val == "B":
                score_b += 1.0

        # Normalize scores (e.g., to be between -0.5 and 0.5)
        total_judgements = len(judgements.choices)
        if total_judgements > 0:
            norm_a = score_a / total_judgements
            norm_b = score_b / total_judgements
            mean_score = (norm_a + norm_b) / 2.0
            score_a = norm_a - mean_score
            score_b = norm_b - mean_score
        else:  # Handle case with no valid judgements
            score_a = 0.0
            score_b = 0.0

        # Tokenize and record the final scored data
        for i, response_content in enumerate(responses):
            chat_history = [
                {"role": "system", "content": character_sheet["system_prompt"]},
                {"role": "user", "content": player_prompt},
                {"role": "assistant", "content": response_content},
            ]
            out_dict = tokenize_for_trainer(self.tokenizer, chat_history)
            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(score_a if i == 0 else score_b)

        return scores

    async def get_next_item(self):
        # Randomly select a character and a player prompt for this turn.
        character_sheet = random.choice(NPC_CHARACTER_SHEETS)
        player_prompt = random.choice(self.player_prompts)

        # The 'item' will now be a dictionary containing the context for this turn.
        item = {
            "character_sheet": character_sheet,
            "player_prompt": player_prompt,
        }
        self.iter += 1
        return item


if __name__ == "__main__":
    NPCArenaEnv.cli()
