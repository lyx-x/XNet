# define the working directory
Root=$1
echo "Loop under $Root."

# define the action script
Action=$2

# list all sub-folders
Folders="$Root*/"

# process all files
for Folder in $Folders
do
	echo $Folder;
	sh $Action $Folder;
done
