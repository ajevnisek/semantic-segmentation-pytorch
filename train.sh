#python train_mask_generator.py   --dataset_name Hday2night_MSE_Adam_bs_16_debug    --img_path ../data/Image_Harmonization_Dataset/Hday2night/composite_images/    --list_path ../data/Image_Harmonization_Dataset/Hday2night/Hday2night_train.txt    --test_list_path ../data/Image_Harmonization_Dataset/Hday2night/Hday2night_test.txt    --mask_path ../data/Image_Harmonization_Dataset/Hday2night/masks/     --target_path ../data/Image_Harmonization_Dataset/Hday2night/real_images/    --gpu_id 0  --n_workers 12 --batch_size 16 --loss MSE  --optimizer Adam    --epoch_size 60  --debug_mode;
python train_mask_generator.py   --dataset_name Hday2night_MSE_Adam_bs_16    --img_path ../data/Image_Harmonization_Dataset/Hday2night/composite_images/    --list_path ../data/Image_Harmonization_Dataset/Hday2night/Hday2night_train.txt    --test_list_path ../data/Image_Harmonization_Dataset/Hday2night/Hday2night_test.txt    --mask_path ../data/Image_Harmonization_Dataset/Hday2night/masks/     --target_path ../data/Image_Harmonization_Dataset/Hday2night/real_images/    --gpu_id 0  --n_workers 12 --batch_size 16 --loss MSE  --optimizer Adam    --epoch_size 60;