
import argparse
import os

import torch
import torch.nn.functional as F
from unet import UNet

def valid_tensor(s):
    msg = "Not a valid resolution: '{0}' [CxHxW].".format(s)
    try:
        q = s.split('x')
        if len(q) != 3:
            raise argparse.ArgumentTypeError(msg)
        return [int(v) for v in q]
    except ValueError:
        raise argparse.ArgumentTypeError(msg)

def parse_args():
    parser = argparse.ArgumentParser(description='UNET exporter')

    parser.add_argument('-c', '--class_count', type=int,
                    help='Class Count', default=5)
    parser.add_argument('-p', '--padding', type=bool,
                    help='apply padding such that the input shape is the same as the output', default=True)
    parser.add_argument('-u', '--up_mode',
                    help="one of 'upconv' or 'upsample", default='upsample')

    parser.add_argument('-r', '--ONNX_resolution', default="1x572x572", type=valid_tensor,
                    help='ONNX input resolution')
    parser.add_argument('-o', '--outfile', default='./out.onnx',
                    help='output file path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    device = torch.device('cpu')
    model = UNet(n_classes=args.class_count, padding=args.padding, up_mode=args.up_mode).to(device)

    # Export ONNX file
    input_names = [ "input:0" ]  # this are our standardized in/out nameing (required for runtime)
    output_names = [ "output:0" ]
    dummy_input = torch.randn([1]+args.ONNX_resolution)
    ONNX_path = args.outfile
    # Exporting -- CAFFE2 compatible
    # requires operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    # https://github.com/pytorch/pytorch/issues/41848
    # for CAFFE2 backend (old exports mode...)
    #torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names, 
    #    keep_initializers_as_inputs=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # Exporting -- ONNX runtime compatible
    #   keep_initializers_as_inputs=True -> is required for onnx optimizer...
    torch.onnx.export(model, dummy_input, ONNX_path, input_names=input_names, output_names=output_names,
        keep_initializers_as_inputs=True, opset_version=11)

if __name__ == '__main__':
    main()
