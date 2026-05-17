# Raw Tensor Semantic Compare

- Image: `/home/ecet400/hailo_detector/test/raccoon_318591236_jpg.rf.czQ4OjoO91jWpAoCqc3F.jpg`
- ONNX: `/home/ecet400/best.onnx`
- HEF: `/home/ecet400/pi_raw_bundle/best_h8l_raw.hef`
- Labels: `bear, coyote, deer, fox, possum, raccoon, skunk, squirrel, turkey`

## Raw Stats
### head0
- ONNX: `{"shape": [1, 3, 80, 80, 14], "min": -26.903722763061523, "max": 4.523876190185547, "mean": -2.939152956008911, "std": 5.00929594039917}`
- HEF: `{"shape": [1, 80, 80, 42], "min": -40.89064407348633, "max": 36.90131378173828, "mean": -4.797345161437988, "std": 5.688204765319824}`
- Best transform: `layout=anchor_major, anchor_perm=[0, 1, 2], corr=0.2063, cosine=0.4622, aligned_rmse=4.9016`

### head1
- ONNX: `{"shape": [1, 3, 40, 40, 14], "min": -23.81147575378418, "max": 4.677489280700684, "mean": -2.6022181510925293, "std": 3.843637704849243}`
- HEF: `{"shape": [1, 40, 40, 42], "min": -16.256616592407227, "max": 4.143843650817871, "mean": -3.117428779602051, "std": 2.5124006271362305}`
- Best transform: `layout=anchor_major, anchor_perm=[2, 1, 0], corr=0.6863, cosine=0.7931, aligned_rmse=2.7956`

### head2
- ONNX: `{"shape": [1, 3, 20, 20, 14], "min": -17.84343719482422, "max": 7.619894981384277, "mean": -2.564537286758423, "std": 3.1890695095062256}`
- HEF: `{"shape": [1, 20, 20, 42], "min": -13.200407981872559, "max": 4.8294172286987305, "mean": -2.975783586502075, "std": 2.6001687049865723}`
- Best transform: `layout=anchor_major, anchor_perm=[1, 0, 2], corr=0.6442, cosine=0.8022, aligned_rmse=2.4391`

## Verdict
HEF raw outputs do not match ONNX raw heads cleanly under simple reshape/transpose/anchor permutations. This points to a deeper semantic mismatch, likely from compile/export or quantization/dequantization behavior.
