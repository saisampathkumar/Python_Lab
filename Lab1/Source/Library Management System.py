import random


class Colorise:
    # Defining class to display text in colors
    def colour(colour, text):
        if colour == "black":
            return "\033[1;30m" + str(text) + "\033[1;m"
        if colour == "red":
            return "\033[1;31m" + str(text) + "\033[1;m"
        if colour == "green":
            return "\033[1;32m" + str(text) + "\033[1;m"
        if colour == "yellow":
            return "\033[1;33m" + str(text) + "\033[1;m"
        if colour == "blue":
            return "\033[1;34m" + str(text) + "\033[1;m"
        if colour == "magenta":
            return "\033[1;35m" + str(text) + "\033[1;m"
        if colour == "cyan":
            return "\033[1;36m" + str(text) + "\033[1;m"
        if colour == "gray":
            return "\033[1;37m" + str(text) + "\033[1;m"
        return str(text)


class Person:
    # Defining person class
    firstName = ""
    lastName = ""

    def __init__(self, firstName, lastName):
        # Defining Constructor
        self.firstName = firstName
        self.lastName = lastName


class Student(Person):
    # Defining Student class
    # Getting Person class inherited to Student class
    iD = int(random.random() * 100000000)
    # Assiging Student Id using random number

    def __init__(self, firstName, lastName):
        Person.__init__(self, firstName, lastName)
    # Defining constructor

    def get_details(self):
        # Printing the Student Details
        print(Colorise.colour("yellow", " Student Id            : "), self.iD)
        print(Colorise.colour("yellow", " First Name            : "), self.firstName)
        print(Colorise.colour("yellow", " Last Name             : "), self.lastName)

    def get_student_name(self):
        # getting the name of the student
        return self.firstName + " " + self.lastName

    def request_Book(self):
        # Defining the function to request book
        self.book = int(input("please enter the Book Number"))
        return self.book

    def return_Book(self):
        # Defining the function to return book
        print(Colorise.colour("blue", "Enter the name of the book to return"))
        self.book = input()
        return self.book


class Library:
    b = []
    # List to save the borrowed books
    r = []
    # List to save the returned books

    def __init__(self, books):
        self.books = books

    def display_books(self):
        # To display available books
        print(":::::::::::::::::::", Colorise.colour("magenta", "BOOK AVAILABLE"), ":::::::::::::::::::")
        for book in self.books:
            print(self.books.index(book) + 1, ".", book)

    def lend_Book(self, rBook):
        # Defining the function to lend a book
        rBook = self.books[rBook - 1]
        print("Thank You for borrowing ", Colorise.colour("cyan", rBook))
        self.books.remove(rBook)
        self.b.append(rBook)

    def return_Book(self, rBook):
        # Defining function to return a book
        self.books.append(rBook)
        print("Thanks for returning ", Colorise.colour("cyan", rBook))
        self.r.append(rBook)

    def get_Summary(self):
        # Defining function to print the total summary of the student
        print("\nBooks which are take")
        for i in self.b:
            print(Colorise.colour("blue", i))
        print("Books which are returned")
        for i in self.r:
            print(Colorise.colour("blue", i))


print("::::::::::::::::::: ", Colorise.colour("magenta", "WELCOME TO LIBRARY"), " :::::::::::::::::::")
print("Please enter your details")
first_Name = input(str("First Name     : "))
last_name = input(str("Last Name      : "))
student = Student(first_Name, last_name)
library = Library(
    ["In Search of Lost Time", "Don Quixote", "Ulysses ", "The Great Gatsby", "Moby Dick", "Hamlet", "War and Peace "])
print("\nWelcome ", student.get_student_name(), "\n")
while True:
    print("\n::::::::::::::::::: ", Colorise.colour("magenta", "LIBRARY MENU"), " :::::::::::::::::::")
    print("""                1. Display all books
                2. Request a book
                3. Return a book
                4. Summary""")
    choice = int(input("Enter Choice:"))
    if choice == 1:
        library.display_books()
    elif choice == 2:
        library.lend_Book(student.request_Book())
    elif choice == 3:
        library.return_Book(student.return_Book())
    elif choice == 4:
        print("\n::::::::::::::::::: SUMMARY :::::::::::::::::::")
        student.get_details()
        library.get_Summary()
        exit()
