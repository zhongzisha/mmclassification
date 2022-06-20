#　rsync -avP  --exclude-from=/usr/exclude.list
for ip in 10.0.7.35 10.0.7.184; do

rsync -av -e ssh \
--exclude='.eggs' --exclude='.git' --exclude='__pycache__' --exclude='.github' --exclude='.idea' --exclude='*.pyc' \
--exclude='mmcls.egg-info' ../mmclassification \
ubuntu@${ip}:/media/ubuntu/Documents/gd/

done