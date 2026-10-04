"""Export real mobile models and compare original/fused/ORT predictions."""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from scripts.export import export_onnx
from zipdepth.course.runtime import load_model, preprocess_bgr, sha256
from zipdepth.utils.model_utils import fuse_remaining_conv_bn


def export_and_verify(checkpoint, sizes, output_dir, prefix, images, atol=1e-4, rtol=1e-3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    original = load_model(checkpoint)
    fused = copy.deepcopy(original)
    fused.fuse_for_inference()
    fuse_remaining_conv_bn(fused)
    results = []
    for size in sizes:
        path = output_dir / f'{prefix}_{size}.onnx'
        export_onnx(copy.deepcopy(fused), (1,3,size,size), str(path))
        onnx.checker.check_model(onnx.load(str(path)))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
        checks = []
        for image_path in images:
            frame = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
            x, _ = preprocess_bgr(frame, size)
            with torch.no_grad():
                ref = original(torch.from_numpy(x)).numpy()
                fused_out = fused(torch.from_numpy(x)).numpy()
            ort_out = session.run(['depth'], {'image':x})[0]
            np.testing.assert_allclose(fused_out, ref, atol=atol, rtol=rtol)
            np.testing.assert_allclose(ort_out, ref, atol=atol, rtol=rtol)
            checks.append({'image':str(image_path), 'fusion_max_abs':float(np.max(np.abs(fused_out-ref))),
                           'onnx_max_abs':float(np.max(np.abs(ort_out-ref)))})
        metadata = {'variant':'base','global_mode':'balanced','upsample_unfold':False,
                    'input':{'name':'image','dtype':'float32','shape':[1,3,size,size],'color':'RGB','range':[0,1],
                             'preprocess':'center_square_crop_then_bilinear_resize','normalization':'inside_model'},
                    'output':{'name':'depth','shape':[1,1,size,size],'units':'relative_inverse_depth_response'},
                    'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'onnx_sha256':sha256(path),
                    'opset':17,'atol':atol,'rtol':rtol,'checks':checks,'verified':True}
        path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append(metadata)
        print(f'VERIFIED {path}: {checks}',flush=True)
    (output_dir / f'{prefix}_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/zipdepth_base_npu.pth')
    p.add_argument('--sizes',type=int,nargs='+',default=[256,384])
    p.add_argument('--output',default='output/models')
    p.add_argument('--prefix',default='zipdepth_baseline')
    p.add_argument('--images',nargs='+',default=['assets/examples/im0.jpg','assets/qualitative/car_show_rgb.jpg'])
    a=p.parse_args()
    if any(s<=0 or s%32 for s in a.sizes): p.error('Sizes must be positive multiples of 32')
    export_and_verify(a.checkpoint,a.sizes,a.output,a.prefix,a.images)

if __name__=='__main__': main()
