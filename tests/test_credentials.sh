#!/bin/bash
# Quick test to verify AWS Bedrock access works

echo "Testing AWS Bedrock access with Llama 4 Maverick..."
echo ""

python -m secdrift.runner \
  --model llama-4-maverick \
  --sectors healthcare \
  --scenarios sql_injection \
  --replicates 1 \
  --output results/credential_test.jsonl

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Credentials work correctly."
    echo ""
    echo "Ready to run full benchmark with:"
    echo "  ./run_optimized_benchmark.sh"
else
    echo ""
    echo "❌ FAILED! Check your AWS credentials and Bedrock access."
fi
