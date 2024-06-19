import os
import argparse
import yaml
from preprocess import preprocess
from DDColor.inference.colorization_pipeline_hf import colorize
from vis import create_video


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_name", type=str, default="blackswan", help="name of the dataset"
    )

    parser.add_argument("--exp_name", type=str, default="base", help="experiment name")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="gpu devices")

    parser.add_argument(
        "--annealed",
        default=True,
        help="whether to apply annealed positional encoding (Only in the warping field)",
    )
    parser.add_argument("--encode_w", default=True, help="whether to apply warping")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="path to the YAML config file.",
    )

    parser.add_argument(
        "--preprocess",
        default=False,
        action="store_true",
        help="whether to convert custom dataset from RGB to Gray",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="train or evaluate or canonical",
        choices=["train", "eval", "canonical", "colorize"],
        required=True,
    )

    args = parser.parse_args()
    args.root_dir = f"../our_dataset/{args.data_name}/{args.data_name}"
    args.mask_dir = f"../our_dataset/{args.data_name}/{args.data_name}_masks_0 ../our_dataset/{args.data_name}/{args.data_name}_masks_1"

    args.log_save_path = f"../logs/our_dataset/{args.data_name}"

    if args.type == "eval" or args.type == "canonical":
        args.test = True
        args.weight_path = (
            f"../ckpts/our_dataset/{args.data_name}/{args.exp_name}/last.ckpt"
        )
        args.canonical_dir = f"../our_dataset/{args.data_name}/{args.exp_name}_control"
        args.flow_dir = None
        args.model_save_path = None
    else:
        args.test = False
        args.weight_path = None
        args.canonical_dir = None
        args.flow_dir = f"../our_dataset/{args.data_name}/{args.data_name}_flow"
        args.model_save_path = f"../ckpts/our_dataset/{args.data_name}"

    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        args_dict = vars(args)
        args_dict.update(config)
        args_new = argparse.Namespace(**args_dict)
        return args_new

    return args


if __name__ == "__main__":
    os.chdir("CoDeF")
    args = get_opts()

    if args.preprocess:
        preprocess(args.root_dir)
        print("Preprocessing done!")

    if args.type == "train":
        # preprocess mask
        if not os.path.exists(args.root_dir + "_masks_0"):
            os.system(
                f"python data_preprocessing/preproc_mask.py --img_dir {args.root_dir} --mask_dir {args.root_dir}_masks"
            )

        # preprocess flow
        if not os.path.exists(args.flow_dir):
            code_dir = "data_preprocessing/RAFT"
            name = args.root_dir.split("/")[-1]
            os.system(
                f"python data_preprocessing/RAFT/demo.py --model {code_dir}/models/raft-sintel.pth \
                                                            --path {args.root_dir} \
                                                            --outdir {args.root_dir}_flow \
                                                            --name {name} \
                                                            --outdir_conf {args.root_dir}_confidence \
                                                            --confidence"
            )

        # train
        os.system(
            f"python train.py --root_dir {args.root_dir} \
                                --model_save_path {args.model_save_path} \
                                --log_save_path {args.log_save_path} \
                                --config {args.config} \
                                --mask_dir {args.mask_dir} \
                                --gpus {args.gpus[0]} \
                                --encode_w --annealed \
                                --exp_name {args.exp_name}"
        )
    elif args.type == "eval":
        os.system(
            f"python train.py --test --encode_w \
                                --root_dir {args.root_dir} \
                                --log_save_path {args.log_save_path} \
                                --weight_path {args.weight_path} \
                                --gpus {args.gpus[0]} \
                                --config {args.config} \
                                --exp_name {args.exp_name}"
        )

        save_dir = os.path.join(
            "..",
            "results",
            args.root_dir.split("/")[1],
            args.root_dir.split("/")[2],
            args.exp_name,
        )
        if not os.path.exists(args.canonical_dir):
            os.makedirs(args.canonical_dir)

        os.system(f"cp {save_dir}/canonical_0.png {args.canonical_dir}/")
    elif args.type == "canonical":
        colorize(args.canonical_dir, args.canonical_dir)
        os.system(
            f"python train.py --test --encode_w \
                                --root_dir {args.root_dir} \
                                --log_save_path {args.log_save_path} \
                                --weight_path {args.weight_path} \
                                --gpus {args.gpus[0]} \
                                --config {args.config} \
                                --exp_name {args.exp_name} \
                                --canonical_dir {args.canonical_dir}"
        )
    elif args.type == "colorize":
        colorize(args.root_dir, f"../DDcolor_output/{args.data_name}")
        create_video(args.data_name)

    os.chdir("..")
