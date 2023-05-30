import argparse
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='Dataset name (e.g., Task001_BrainCancerClassification)')
    args, unknown = parser.parse_known_args()

    plan_and_preprocess_entry(args.dataset_name, unknown)
