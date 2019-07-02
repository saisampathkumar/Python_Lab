import random

class colorise:

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
    firstName = ""
    lastName = ""
    dOB = ""
    Address = ""
    cntct = ""
    email = ""

    def __init__(self, firstName, lastName, dOB, Address, cntct, email):
        self.firstName = firstName
        self.lastName = lastName
        self.dOB = dOB
        self.Address = Address
        self.cntct = cntct
        self.email = email


class Passenger(Person):
    iD = int(random.random() * 100000000)

    def __init__(self, firstName, lastName, dOB, Address, cntct, email):
        Person.__init__(self, firstName, lastName, dOB, Address, cntct, email)

    def get_details(self):
        print(" |    ", colorise.colour("green", " Passenger Id          : "), self.iD, "  |")
        print(" |    ", colorise.colour("green", " First Name            : "), self.firstName, "   |   ")
        print(" |    ", colorise.colour("green", " Last Name             : "), self.lastName, "   |   ")
        print(" |    ", colorise.colour("green", " Address               : "), self.Address, "   |   ")
        print(" |    ", colorise.colour("green", " Email                 : "), self.email, "   |   ")
        print(" |    ", colorise.colour("green", " Phone                 : "), self.cntct, "   |   ")
        print(" |    ", colorise.colour("green", " Date Of Birth         : "), self.dOB, "   |   ")


class Check_Schedule(colorise):
    data = [["May 22nd", "May 23rd", "12 hrs", "$ 137", 35, 100, 1979898],
            ["May 24th", "May 25th", "15 hrs", "$ 98", 90, 150, 1987789],
            ["May 23rd", "May 24th", "15.5 hrs", "$ 90", 87, 100, 1928939],
            ["May 22nd", "May 23rd", "13 hrs", "$ 129", 20, 80, 1282939],
            ["May 23rd", "May 24th", "17.7 hrs", "$ 80", 110, 200, 1282901]]

    def __init__(self, loc1, loc2, pas, flight_id, firstName, lastName, dOB, Address, cntct, email):
        self.loc1 = loc1
        self.loc2 = loc2
        self.pas = pas
        self.flight_id = flight_id
        self.firstName = firstName
        self.lastName = lastName
        self.dOB = dOB
        self.Address = Address
        self.cntct = cntct
        self.email = email

    def get_flight_details(self):
        print(colorise.colour("blue", " Departure Location :"), self.loc1, "\n",
              colorise.colour("blue", "Destination        :"), self.loc2, "\n")
        for data in Check_Schedule.data:
            print(colorise.colour("yellow", "Flight Id         : "), data[6])
            print(colorise.colour("yellow", "Departure Date    : "), data[0])
            print(colorise.colour("yellow", "Arrival Date      : "), data[1])
            print(colorise.colour("yellow", "Duration          : "), data[2])
            print(colorise.colour("yellow", "Cost              : "), data[3])
            print(colorise.colour("yellow", "Available Seats   : "), data[4], "\n")

    @property
    def update_flight_details(self):
        for data in Check_Schedule.data:
            if data[6] == self.flight_id:
                if data[4] == 0:
                    print("No seats available")
                    return False
                else:
                    res = input(str("Seats are available Do you want to confirm the ticket: Y/N "))
                    if res == "Y":
                        data[4] = data[4] - 1
                        print("Your Ticket is Confirmed")
                        return True
                    elif res == "N":
                        return False

    def print_ticket(self):
        print(":::::::::::::::::::::::::::::Ticket:::::::::::::::::::::::::::::")
        print("----------------------------------------------------------------")
        print(" |    ", colorise.colour("green", " Departure Location    : "), self.loc1, "  |")
        print(" |    ", colorise.colour("green", " Destination           : "), self.loc2, "  |")
        for data in Check_Schedule.data:
            if data[6] == self.flight_id:
                p = Passenger(self.firstName, self.lastName, self.dOB, self.Address, self.cntct, self.email)
                p.get_details()
                r = int(random.random() * 100000000)
                print(" |    ", colorise.colour("green", "Ticket Confirmation    : "), r, "  |")
                print(" |    ", colorise.colour("green", "Flight Id              : "), data[6], "  |")
                print(" |    ", colorise.colour("green", "Departure Date         : "), data[0], "  |")
                print(" |    ", colorise.colour("green", "Arrival Date           : "), data[1], "  |")
                print(" |    ", colorise.colour("green", "Duration               : "), data[2], "  |")
                print(" |    ", colorise.colour("green", "Cost                   : "), data[3], "  |")
                print(" |    ", colorise.colour("green", "Available Seats        : "), data[4], "   |", "\n")
                print("----------------------------------------------------------------")


class Issue_ticket():

    def __init__(self, loc1, loc2, pas, flight_id, firstName, lastName, dOB, Address, cntct, email):
        self.loc1 = loc1
        self.loc2 = loc2
        self.pas = pas
        self.flight_id = flight_id
        self.firstName = firstName
        self.lastName = lastName
        self.dOB = dOB
        self.Address = Address
        self.cntct = cntct
        self.email = email

    def check_status(self):
        status = Check_Schedule(self.loc1, self.loc2, self.pas, self.flight_id, self.firstName, self.lastName, self.dOB,
                                self.Address, self.cntct, self.email)
        res = status.update_flight_details
        if res:
            status.print_ticket()
        elif not res:
            print("Something went wrong please try again later")


class Reservation_System():
    print("\n ::::::::::::::::::::::::::Welcome to STUDENT AIRLINES::::::::::::::::::::::::::")
    loc1 = input(str("Please Enter Departure Location: "))
    loc2 = input(str("Please Enter Arrival Location: "))
    pas = input("How many passengers: ")
    print("Please enter your details")
    first_Name = input(str("First Name     : "))
    last_name = input(str("Last Name      : "))
    address = input(str("Address        : "))
    email = input(str("Email          : "))
    phone = input(str("Phone          : "))
    dOB = input(str("Date Of Birth  : "))
    print("\n::::::::::::::::::::::::::Flight Details::::::::::::::::::::::::::")
    tic = Check_Schedule(loc1, loc2, pas, 1979898, first_Name, last_name, address, email, phone, dOB)
    tic.get_flight_details()
    f_id = int(input("\n Please enter flight ID to reserve a seat for you: "))
    tic1 = Issue_ticket(loc1, loc2, pas, f_id, first_Name, last_name, address, email, phone, dOB)
    tic1.check_status()


r1 = Reservation_System()

