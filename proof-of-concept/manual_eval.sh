for photo in manual_eval/*; do
  python predict.py "$photo"
  echo 
done
