import os

places = 4

suffix = "img"
extension = "jpg"

previous_job = 0
last_number = 0

for filename in os.listdir("."):
    
    if(suffix + " (" in filename):
        newname = filename.replace(" (", "_", 1)
        newname = newname.replace(")", "", 1)

        num = filename.replace(suffix + " (", "", 1).replace(")." + extension, "", 1)
        
        if(len(filename) < len(suffix + " ()." + extension) + places):#pad the number
            
            orig_num = str(num)
            
            num = "0"*((len(suffix + " ()." + extension) + places) - len(filename)) + num

            newname = newname.replace(orig_num, num)

        last_number = int(num)
            
        os.rename(filename, newname)
        
        
    elif(suffix + "_" in filename):
        num = filename.replace(suffix + "_", "", 1).replace("." + extension, "", 1)

        if(len(filename) < len(suffix + "_." + extension) + places):#pad the number
            orig_num = str(num)
            
            num = "0"*((len(suffix + "_." + extension) + places) - len(filename)) + num

            newname = filename.replace(orig_num, num)

            os.rename(filename, newname)

        last_number = int(num)
        
    elif(".thumb" in filename):
        last_number += 1

        padded = "0"*(places - len(str(last_number))) + str(last_number)
        
        os.rename(filename, str(suffix + "_%d." + extension) % (last_number))
        
    if(last_number != previous_job):
        print("worked on file %d" % last_number)
        previous_job = last_number

print("finished all jobs")
