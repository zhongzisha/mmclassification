time1=`date`              # 获取当前时间
time2=$(date -d "-40 minute ago" +"%Y-%m-%d %H:%M:%S")  # 获取两个小时后的时间

t1=`date -d "$time1" +%s`     # 时间转换成timestamp
t2=`date -d "$time2" +%s`

echo t1=$t1
echo t2=$t2

while [ $t1 -lt $t2 ]     # 循环，不断检查是否来到了未来时间
do
  echo "wait for 60 seconds .."
  sleep 60
  time1=`date`
  t1=`date -d "$time1" +%s`
  echo t1=$t1
done

echo "yes"       # 循环结束，开始执行任务
echo $time1
echo $time2

sleep 60

CONFIG=resnet101_b32x2_ganta_with_tower_state
./tools/dist_train.sh configs/resnet/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}

sleep 60

CONFIG=resnext50_32x4d_b32x2_ganta_with_tower_state
./tools/dist_train.sh configs/resnext/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}

CONFIG=regnetx_3.2gf_b32x2_ganta_with_tower_state
./tools/dist_train.sh configs/regnet/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}


CONFIG=seresnet50_b32x2_ganta_with_tower_state
./tools/dist_train.sh configs/seresnet/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}

CONFIG=seresnext50_32x4d_b32x2_ganta_with_tower_state
./tools/dist_train.sh configs/seresnext/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}

CONFIG=swin_base_224_b32x2_300e_ganta_with_tower_state
./tools/dist_train.sh configs/swin_transformer/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}


CONFIG=mobilenet_v3_large_ganta_with_tower_state
./tools/dist_train.sh configs/mobilenet_v3/${CONFIG}.py 2 --work-dir /media/ubuntu/Temp/ganta_with_tower_state/${CONFIG}
