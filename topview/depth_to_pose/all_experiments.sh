python main.py --train_dataset ITOP --validation_dataset ITOP --test_dataset ITOP --epochs 15 --experiment_code itop_itop_itop
python main.py --train_dataset ITOP --validation_dataset ITOP --test_dataset PANOPTIC --epochs 15 --experiment_code itop_itop_panoptic
python main.py --train_dataset ITOP --validation_dataset BOTH --test_dataset PANOPTIC --epochs 15 --experiment_code itop_both_panoptic
python main.py --train_dataset PANOPTIC --validation_dataset PANOPTIC --test_dataset PANOPTIC --epochs 15 --experiment_code panoptic_panoptic_panoptic
