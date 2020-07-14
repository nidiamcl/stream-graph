#!/bin/sh
# python get_good_clusters.py --n 'virgili' --t1 0.1111111111111111 --t2 0.1111111111111111
# python get_good_clusters.py --n 'zachary_karate_club' --t1 0.2222222222222222 --t2 0.3333333333333333
# python get_good_clusters.py --n 'zebra' --t1 0.3333333333333333 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveyaug17' --t1 0.3333333333333333 --t2 0.9
# python get_good_clusters.py --n 'harveyaug18' --t1 0.2222222222222222 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveyaug19' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveyaug23' --t1 0.3333333333333333 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveyaug24' --t1 0.1111111111111111 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveyaug26' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveyaug30' --t1 0.2222222222222222 --t2 0.9
# python get_good_clusters.py --n 'harveysept01' --t1 0.3333333333333333 --t2 0.9
# python get_good_clusters.py --n 'harveysept02' --t1 0.2222222222222222 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveysept03' --t1 0.3333333333333333 --t2 0.5555555555555556
# python get_good_clusters.py --n 'harveysept04' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept05' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept06' --t1 0.3333333333333333 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept07' --t1 0.4444444444444444 --t2 0.9
# python get_good_clusters.py --n 'harveysept08' --t1 0.4444444444444444 --t2 0.9
# python get_good_clusters.py --n 'harveysept09' --t1 0.4444444444444444 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveysept10' --t1 0.3333333333333333 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept11' --t1 0.3333333333333333 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept12' --t1 0.2222222222222222 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept13' --t1 0.1111111111111111 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept14' --t1 0.1111111111111111 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept15' --t1 0.3333333333333333 --t2 0.9
# python get_good_clusters.py --n 'harveysept16' --t1 0.1111111111111111 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept17' --t1 0.2222222222222222 --t2 0.8888888888888888
# python get_good_clusters.py --n 'harveysept18' --t1 0.2222222222222222 --t2 0.9
# python get_good_clusters.py --n 'harveysept19' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept20' --t1 0.2222222222222222 --t2 0.9
# python get_good_clusters.py --n 'harveysept21' --t1 0.2222222222222222 --t2 0.7777777777777777
# python get_good_clusters.py --n 'harveysept22' --t1 0.1111111111111111 --t2 0.6666666666666666
# python get_good_clusters.py --n 'harveysept23' --t1 0.3333333333333333 --t2 0.9
# python get_good_clusters.py --n 'harveysept24' --t1 0.1111111111111111 --t2 0.9
# python get_good_clusters.py --n 'harveysept25' --t1 0.2222222222222222 --t2 0.9
# python get_good_clusters.py --n 'bible' --t1 0.2222222222222222 --t2 0.1111111111111111
# python get_good_clusters.py --n 'caenorhabditis_elegans' --t1 0.1111111111111111 --t2 0.3333333333333333
# python get_good_clusters.py --n 'chicago' --t1 0.9 --t2 0.9
# python get_good_clusters.py --n 'contiguous-usa' --t1 0.1111111111111111 --t2 0.9
# python get_good_clusters.py --n 'david_coperfield' --t1 0.1111111111111111 --t2 0.3333333333333333
# python get_good_clusters.py --n 'dnc-corecipient' --t1 0.5555555555555556 --t2 0.2222222222222222
# python get_good_clusters.py --n 'dolphins' --t1 0.2222222222222222 --t2 0.1111111111111111
# python get_good_clusters.py --n 'euroroad' --t1 0.1111111111111111 --t2 0.1111111111111111
# python get_good_clusters.py --n 'facebook_NIPS' --t1 0.1111111111111111 --t2 0.7777777777777777
# python get_good_clusters.py --n 'friendships-hamster' --t1 0.1111111111111111 --t2 0.1111111111111111
# python get_good_clusters.py --n 'infectious' --t1 0.1111111111111111 --t2 0.1111111111111111
# python get_good_clusters.py --n 'les_miserables' --t1 0.5555555555555556 --t2 0.1111111111111111
# python get_good_clusters.py --n 'PDZBase' --t1 0.9 --t2 0.5555555555555556
# python get_good_clusters.py --n 'protein' --t1 0.1111111111111111 --t2 0.1111111111111111
# python get_good_clusters.py --n 'train_bombing'  --t1 0.5555555555555556 --t2 0.5555555555555556
# python get_good_clusters.py --n 'virgili'  --t1 0.1111111111111111 --t2 0.1111111111111111


# chmod +x run_serial.sh
# ./run_serial.sh &> execution_times_serial.txt
