import os

def Create_File():
    filename = str(input("Enter the filename to create: "))
    file = open(filename, 'x')
    
def Read_File():
    filename = str(input("Enter the filename to read: "))
    file = open(filename, 'r')
    print(f"Content in file: {file.read()}")
    file.close()
    
def Edit_File():
    filename = str(input("Enter the filename to edit: "))
    file = open(filename, 'w')
    cont =  str(input("Enter the Content: "))
    write = file.write(cont)
    print(f"Content written: {write}")
    
def Delete_File():
    filename = str(input("Enter the filename to delete: "))
    file = os.remove(filename)
    print(f"deleted: {filename}")
    
def view_file():
    files = os.listdir()
    for f in files:
        print(f"Files in Directory: {files}")
        

# Choices

print("1. Create a new File")
print("2. Read a File")
print("3. Edit a File")
print("4. Delete a File")
print("5. View all Files on Directory")

Input = int(input("Enter Opration choice to Perform: "))

match Input:
    
    case 1:
        Create_File()
    
    case 2:
        Read_File()
        
    case 3:
        Edit_File()
        
    case 4:
        Delete_File()
        
    case 5:
        view_file()