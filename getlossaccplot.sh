pushd val
for method in {eye,enet,wlasso,wridge,owl,lasso,ridge}
do
    echo "$method"
    mkdir -p "montage_${method}"
    for i in "${method}"*
    do
	echo $i
	cp "${i}/loss_plot.png" "montage_${method}/loss_${i}.png"
	cp "${i}/acc_plot.png" "montage_${method}/acc_${i}.png"	
    done
    pushd "montage_${method}"
    montage loss* -geometry +2+2 montage_loss.png
    montage acc* -geometry +2+2 montage_acc.png
    rm loss*
    rm acc*
    popd
done
popd
