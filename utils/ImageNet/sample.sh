# define the working directory
Root=$1
echo "Sampling under $Root"

Dest=$3
rm -f $Dest;

# read category list
Cat=$2
while read Folder
do
	echo $Folder;

	# list all files
	Files="$Root""n$Folder/*.JPEG";

	# process all files
	for File in $Files
	do
		File=$(echo $File | cut -c35-);
		Label=$(echo $File | cut -c-8);
		Num=$(echo $File | cut -c10- | awk -F . '{ print $1 }');
		echo "$Label $Num" >> $Dest;
	done
done < $Cat
