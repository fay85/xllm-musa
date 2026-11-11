# Packed FP4 BF16-halfway regression

Synthetic input and packed E2M1/UE8M0 row extracted from the deterministic GPU seed 111107 in validate_decode_moe.py. No model/customer data. Original projection N=2304,K=5120, local expert0, up-row147. Naive FP32 accumulation rounded the BF16 result to0.1953125; independent FP64 dot and muDNN BF16 gave0.1943359375. This caused one composite output to fail the existing rtol=1/64,atol=5e-4 check. FP32 compensated accumulation fixes the projection without relaxing that gate or adding CPU/FP64 runtime math.

Run tools/musa_deepseek_v41/validate_decode_moe.py --output result.json on one idle GPU with the matching test extension on PYTHONPATH. The default numeric fixture is this file. The loader uses torch.load(weights_only=True).
