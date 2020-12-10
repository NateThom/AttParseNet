import argparse

def get_args():
    parser = argparse.ArgumentParser(description='This script loads or trains the CNN.')

    parser.add_argument('--image_path',
                        default='/home/nthom/Documents/datasets/CelebA/Img/resized_images_178x218',
                        help='Path to input data directory [/home/user/Documents/input_images/]')

    parser.add_argument('--attr_label_path',
                        default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/list_attr_celeba_attparsenet.csv',
                        help='Path to mapping between input images and binary attribute labels [/home/user/Documents/list_attr_celeba_attparsenet.csv]')

    parser.add_argument('--mask_image_path',
                        default='/home/nthom/Documents/datasets/CelebA/Img/segment_labels_178x218',
                        help='Path to segment label data [/home/user/Documents/attribute_segment_labels/]')

    parser.add_argument('--mask_label_path',
                        default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/segment_labels.csv',
                        help='Path to mapping between input images and segment labels= [home/Documents/segment_labels.csv]')

    parser.add_argument('--metrics_output_path',
                        default='/home/nthom/Documents/attparsenet_data/output_metrics/',
                        help='File for saving metrics [/home/user/Documents/metrics/metric.txt]')

    parser.add_argument('--metrics_csv_output_path',
                        default='/home/nthom/Documents/attparsenet_data/csv_output_metrics/',
                        help='File for saving metrics in csv format [/home/user/Documents/metrics/metric.csv]')

    parser.add_argument('--model',
                        default="attparsenet",
                        # default="vgg16",
                        # default="moon",
                        help='Designates the model to be initialized [attparsenet]')

    parser.add_argument('--load',
                        default=False,
                        help='True for loading a pretrained model, False otherwise [0]')

    parser.add_argument('--load_path',
                        default='/home/nthom/Documents/attparsenet_data/models/',
                        help='File path for the model to load [/home/user/Document/models/]')

    parser.add_argument('--load_file',
                        default='/epoch_9_loss_738.7768128812313',
                        help='File name for the model to load [/model_to_load]')

    parser.add_argument('--save',
                        default=True,
                        help='True for saving the model, False otherwise [True]')

    parser.add_argument('--save_path',
                        default='/home/nthom/Documents/attparsenet_data/models/',
                        help='Dir for saving models [./saved_models/]')

    parser.add_argument('--train_by_num_epoch',
                        default=True,
                        help='True for training by preset number of epochs')

    parser.add_argument('--train_epochs',
                        default=24,
                        help='Number of training epochs [22]')

    parser.add_argument('--train_by_comparison_with_validation',
                        default=False,
                        help='True for training by comparison with loss on validation set')

    parser.add_argument('--validate',
                        default=False,
                        help='True for evaluation on the validation set')

    parser.add_argument('--test',
                        default=False,
                        help='True for evaluation on the test set')

    parser.add_argument('--train_size',
                        #default=1000,
                        default=160000,
                        help='Number of samples in training set [162770]')

    parser.add_argument('--val_size',
                        default=20000,
                        help='Number of samples in validation set [19867]')

    parser.add_argument('--test_size',
                        default=20000,
                        help='Number of samples in test set [19963]')

    parser.add_argument('--all_size',
                        default=200000,
                        help='Total Number of samples in the dataset [202600]')

    parser.add_argument('--balance',
                        default=False,
                        help='Check the batch and reweight samples for balancing [True]')

    parser.add_argument('--segment',
                        default=True,
                        help='Train with segmentation (mse) loss in addition to bce loss [True]')

    parser.add_argument('--shuffle',
                        default=True,
                        help='Shuffle the order of training samples. Validation and Testing sets will not be shuffled [True]')

    parser.add_argument('--random_seed',
                        default=64,
                        help='Seed for random number generators [64]')

    parser.add_argument('--parallelize',
                        default=True,
                        help='If True, parallelize the model across multiple devices (usually GPUs) [True]')

    parser.add_argument('--batch_size',
                        default=68,
                        help='Batch size for images [32]')

    parser.add_argument('--lr',
                        default=0.001,
                        help='Learning rate [0.001]')

    parser.add_argument('--plot_loss',
                        default=False,
                        help='If True, plot loss curve across all epochs [False]')

    parser.add_argument('--show_parameters',
                        default=False,
                        help='If True, show network parameters [False]')

    parser.add_argument('--attr_list',
                        default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                'Eyeglasses', 'Goatee', 'Grey_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Side_Burns',
                                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
                        help='List of all 40 attributes')

    return parser.parse_args()
