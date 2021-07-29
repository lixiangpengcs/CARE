


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Mutli-View Attention Networks")
  arg_parser.add_argument("--model", dest="model", type=str, default='mvan',  # mvan, base, rva, transformer
                          help="Model Name")
  arg_parser.add_argument("--version", dest="version", type=str, default='0.9', # 1.0, 0.9
                          help="Dataset Version")
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str,
                          help="Evaluation Checkpoint", default=False)
  arg_parser.add_argument("--eval_split", dest="eval_split", type=str,
                          help="Evaluation split", default="test")
  arg_parser.add_argument("--description", dest="description", type=str, default='',
                          help="Image Region Proposal Type")
  arg_parser.add_argument("--eval_seed", dest="eval_seed", type=str,
                          help="Evaluation split", default="3143")

  args = arg_parser.parse_args()
  if args.evaluate:
    print("EVALUATION")
    evaluate(args)
  else:
    print("MODEL TRAIN")
    train_model(args)