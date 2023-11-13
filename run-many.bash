for (( i=0; i<11; i++ ))
do
    for (( j=0; j<33; j++ ))
    do
        python neighbors.py -m run -e many -f generated-configs-many/custom-$i/$j -R results_$1 -r 1000
    done
done

