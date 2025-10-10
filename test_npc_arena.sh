#!/bin/bash

# Test script for the NPC Arena Environment
# This script tests the environment's ability to generate and score NPC dialogue.

set -e  # Exit on error

echo "=========================================="
echo "NPC Arena Environment Test Script"
echo "=========================================="
echo ""

# Step 1: Check for OpenAI API key
echo "Step 1: Checking OpenAI API configuration..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "This test uses an OpenAI model (gpt-4o-mini) as both the trainee and the judge."
    echo "Please set the key with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi
echo "✓ OpenAI API key found"
echo ""

# Step 2: Run a test with the NPC Arena environment
echo "Step 2: Running test with npc_arena_server.py..."
echo "This will process a small number of rollouts to ensure the environment runs."
echo ""

# Configuration
OUTPUT_FILE="npc_arena_test_output_$(date +%Y%m%d_%H%M%S).jsonl"
GROUP_SIZE=2
TOTAL_STEPS=2 # Run just a couple of steps for a quick test

echo "Configuration:"
echo "  - Model: gpt-4o-mini (OpenAI)"
echo "  - Output file: $OUTPUT_FILE"
echo "  - Group size: $GROUP_SIZE"
echo "  - Total steps: $TOTAL_STEPS"
echo ""

# Run the process command with test settings
# Note: We are running this as a module to ensure correct pathing.
# We override several config settings for this test run.
python -m environments.npc_arena_server process \
    --env.data_path_to_save_groups "$OUTPUT_FILE" \
    --env.group_size $GROUP_SIZE \
    --env.total_steps $TOTAL_STEPS \
    --env.use_wandb false \
    --openai.model_name "gpt-4o-mini" \
    --openai.base_url "https://api.openai.com/v1" \
    --openai.api_key "$OPENAI_API_KEY"

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files created:"
    echo "  - $OUTPUT_FILE (JSONL data)"
    echo "  - ${OUTPUT_FILE%.jsonl}.html (HTML visualization)"
    echo ""
    echo "You can open the HTML file in a browser to view the generated dialogues and judgements."
else
    echo ""
    echo "ERROR: Test failed!"
    exit 1
fi