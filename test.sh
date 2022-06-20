SAVE_ROOT=/media/ubuntu/Temp/ganta_with_tower_state

# shellcheck disable=SC2045
for CONFIG in `ls $SAVE_ROOT`; do
echo $CONFIG
if [ ! -e "$SAVE_ROOT/$CONFIG/report.txt" ]; then

  python tools/test.py \
  $SAVE_ROOT/$CONFIG/$CONFIG.py \
  $SAVE_ROOT/$CONFIG/latest.pth \
  --metrics accuracy \
  --out $SAVE_ROOT/$CONFIG/results.pkl


  python tools/analysis_tools/analyze_results_v2.py \
  $SAVE_ROOT/$CONFIG/$CONFIG.py \
  $SAVE_ROOT/$CONFIG/results.pkl > $SAVE_ROOT/$CONFIG/report.txt

else

  cat $SAVE_ROOT/$CONFIG/report.txt

fi

done












