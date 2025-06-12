# run this in dataset folder instead of here

colmap feature_extractor \
    --database_path database.db \
    --image_path images/ 
    

colmap exhaustive_matcher \
    --database_path database.db 


colmap mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path sparse/ \
    --Mapper.num_threads 8 

colmap model_converter \
    --input_path sparse/0/ \
    --output_path sparse/0/ \
    --output_type TXT

colmap image_undistorter \
    --image_path images/ \
    --input_path sparse/0/ \
    --output_path undistorted/
    
# mkdir undistorted/sparse/0/
# put all the undistorted/sparse/ into undistorted/sparse/0/

colmap model_converter \
    --input_path undistorted/sparse/0/ \
    --output_path undistorted/sparse/0/ \
    --output_type TXT
