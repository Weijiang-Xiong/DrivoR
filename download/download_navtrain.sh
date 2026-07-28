hf download OpenDriveLab/OpenScene openscene-v1.1/openscene_metadata_trainval.tgz --repo-type dataset --local-dir ./
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz
mv openscene-v1.1/meta_datas trainval_navsim_logs
rm -r openscene-v1.1

mkdir -p trainval_sensor_blobs/trainval
for split in {1..32}; do
    hf download OpenDriveLab/OpenScene navsim/navtrain_current_${split}.tgz --repo-type dataset --local-dir ./
    mv navsim/navtrain_current_${split}.tgz .
    echo "Extracting file navtrain_current_${split}.tgz"
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz

    rsync -r navtrain_current_${split}/* trainval_sensor_blobs/trainval
    rm -rf navtrain_current_${split}
done

for split in {1..32}; do
    hf download OpenDriveLab/OpenScene navsim/navtrain_history_${split}.tgz --repo-type dataset --local-dir ./
    mv navsim/navtrain_history_${split}.tgz .
    echo "Extracting file navtrain_history_${split}.tgz"
    tar -xzf navtrain_history_${split}.tgz
    rm navtrain_history_${split}.tgz

    rsync -rv navtrain_history_${split}/* trainval_sensor_blobs/trainval
    rm -r navtrain_history_${split}
done