# define the working directory
Folder=$1
echo "Resizing under $Folder."

# list all files
Files="$Folder*"

# process all files
for File in $Files
do
	# echo $File;
	cp $File "$File.bak";
	convert $File -resize 256x256^ -gravity center -extent 256x256^ $File;
done
