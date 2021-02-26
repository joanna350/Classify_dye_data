mkdir -p data plots

if ls *.csv &>/dev/null
then
    mv *.csv data/
fi
