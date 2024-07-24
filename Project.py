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