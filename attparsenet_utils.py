import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path',
                        default='/home/nthom/Documents/datasets/CelebA/Img/',
                        # default='/home/nthom/Documents/datasets/CelebA/Img/partial_blackout/',
                        #default='/home/nthom/Documents/datasets/lfwa/',
                        # default='/home/nthom/Documents/datasets/UMD-AED/',
                        help='Path to input data directory [/home/user/Documents/input_images/]')

    parser.add_argument('--image_dir',
                        default='resized_images_178x218',
                        # default='resized_aligned_images_178x218',
                        # default='resized_segme*nt1',
                        # default='lfw',
                        # default='croppedImages',
                        help='input_images')

    parser.add_argument('--attr_label_path',
                        # default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/list_attr_celeba_attparsenet.csv',
                        default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/list_attr_celeba_attparsenet_adjusted.csv',
                        # default='/home/nthom/Documents/datasets/lfwa/lfwa_labels_full_paths.csv',
                        # default='/home/nthom/Documents/datasets/UMD-AED/Files_attparsenet/list_attr_umdaed_reordered.csv',
                        help='Path to mapping between input images and binary attribute labels [/home/user/Documents/list_attr_celeba_attparsenet.csv]')

    parser.add_argument('--mask_image_path',
                        default='/home/nthom/Documents/datasets/CelebA/Img/segment_labels_178x218',
                        help='Path to segment label data [/home/user/Documents/attribute_segment_labels/]')

    parser.add_argument('--mask_label_path',
                        # default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/segment_labels.csv',
                        default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/segment_labels_adjusted.csv',
                        help='Path to mapping between input images and segment labels= [home/Documents/segment_labels.csv]')

    parser.add_argument('--metrics_output_path',
                        default='/home/nthom/Documents/attparsenet_data/output_metrics/',
                        help='File for saving metrics [/home/user/Documents/metrics/metric.txt]')

    parser.add_argument('--metrics_csv_output_path',
                        default='/home/nthom/Documents/attparsenet_data/csv_output_metrics/',
                        help='File for saving metrics in csv format [/home/user/Documents/metrics/metric.csv]')

    parser.add_argument('--model',
                        default="attparsenet",
                        help='Designates the model to be initialized [attparsenet]')

    parser.add_argument('--load',
                        default=False,
                        # default=True,
                        help='True for loading a pretrained model, False otherwise [0]')

    parser.add_argument('--load_path',
                        default='/home/nthom/Documents/AttParseNet/checkpoints/',
                        help='File path for the model to load [/home/user/Document/models/]')

    parser.add_argument('--load_file',
                        # Aligned Baseline
                        default='baseline_mult_hflip/Baseline_Aligned_mult_hflip_40_0.01_Mworks06-epoch=17-Validation Loss=0.21172.ckpt',
                        #AttParseNet
                        # default='AttParseNet_Unaligned_Mult/AttParseNet_Unaligned_mult_40_0.01_Mworks08-epoch=16-Validation Loss=0.30774.ckpt',
                        # default='AttParseNet_Unaligned_40_0.01_Mworks08-epoch=49-Validation Loss=0.33161.ckpt',
                        # default='AttParseNet_Mult_hflip/AttParseNet_Unaligned_mult_hflip_40_0.01_Mworks08-epoch=21-Validation Loss=0.33261.ckpt',
                        # default='AttParseNet_Mult_hflip/AttParseNet_Unaligned_mult_hflip_40_0.01_Mworks08-epoch=22-Validation Loss=0.33261.ckpt',
                        help='File name for the model to load [/model_to_load]')

    parser.add_argument('--save',
                        default=True,
                        # default=False,
                        help='True for saving the model, False otherwise [True]')

    parser.add_argument('--save_path',
                        default='/home/nthom/Documents/AttParseNet/checkpoints',
                        help='Dir for saving models [./saved_models/]')

    parser.add_argument('--train_epochs',
                        default=50,
                        help='Number of training epochs [22]')

    parser.add_argument('--train_size',
                        #lfwa and umd
                        # default=0,
                        #celeba
                        default=160000,
                        help='Number of samples in training set [162770]')

    parser.add_argument('--val_size',
                        #lfwa and umd
                        # default=0,
                        #celaba
                        default=20000,
                        help='Number of samples in validation set [19867]')

    parser.add_argument('--test_size',
                        #lfwa
                        # default=13088,
                        # umd
                        # default=2808,
                        #celeba
                        default=20000,
                        help='Number of samples in test set [19963]')

    parser.add_argument('--all_size',
                        #lfwa
                        # default=13088,
                        # umd
                        # default=2808,
                        #celeba
                        default=200000,
                        help='Total Number of samples in the dataset [202600]')

    parser.add_argument('--train',
                        # default=False,
                        default=True,
                        help='Train the model on the training set and evaluate on the validation set')

    parser.add_argument('--val_only',
                        default=False,
                        # default=True,
                        help='Evaluate the model on the validation set')

    parser.add_argument('--test',
                        default=False,
                        # default=True,
                        help='Evaluate the model on the test set')

    parser.add_argument('--save_feature_maps',
                        default=False,
                        # default=True,
                        help='Save all feature maps for data in either the test or val set')

    parser.add_argument('--show_batch',
                        default=False,
                        # default=True,
                        help='Show the batch input images and masks for debugging')

    parser.add_argument('--repair_labels',
                        # default=False,
                        default=True,
                        help='Show the batch input images and masks for debugging')

    parser.add_argument('--segment',
                        # default=False,
                        default=True,
                        help='Train with segmentation (mse) loss in addition to bce loss [True]')

    parser.add_argument('--shuffle',
                        # default=False,
                        default=True,
                        help='Shuffle the order of training samples. Validation and Testing sets will not be shuffled [True]')

    parser.add_argument('--random_seed',
                        default=64,
                        help='Seed for random number generators [64]')

    parser.add_argument('--batch_size',
                        default=40,
                        help='Batch size for images [32]')

    parser.add_argument('--lr',
                        default=0.01,
                        help='Learning rate [0.001]')

    parser.add_argument('--patience',
                        default=5,
                        help='Learning Rate Scheduler Patience [5]')

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
