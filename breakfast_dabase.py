#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:55:58 2021

@author: sabal
"""


import sys
import random
import os 
import time


class Objects:
    def __init__(self):
        self.cup = ["blue cup","black cup"]
        self.sugar = ["glass tin of brown sugar", "brown sugar container", "sachets of white sugar", "paper bag of white sugar", "plastic tin of white sugar"]
        self.milk = ["whole milk", "semi-skimmed milk"]
        self.spoon = ["big spoon", "small spoon"]
        self.knife = ["grey knife", "black knife"]
        self.plate = ["big plate", "small plate"]
        self.sliced_bread = ["tin of sliced bread", "slice of sliced bread"]
        self.jam = ["strawberry jam","blackberry jam", "apricot jam"]
        self.cereals = ["frosties cereals", "corn flakes cereals", "choco krispies cereals"]
        self.bowl = ["white bowl", "blue bowl"]
        self.decision = []
        
    def prepare_drink (self):

        option_selected = random.randint(0, 3)
        sugar_decision = random.randint(0, 1)
        cup_selected = random.randint(0, 1)

        if option_selected == 0: 

            self.decision = ["COFFEE BREWING","water", "coffee", self.cup[cup_selected], self.spoon[1]]

        else:
            milk_selected = random.randint(0, 1)
            
            if option_selected == 1: 
                self.decision = ["COFFEE BREWING",self.milk[milk_selected], "coffee", self.cup[cup_selected], self.spoon[1]]
            elif option_selected == 2: 
                self.decision = ["NESQUICK BREWING",self.milk[milk_selected], "nesquick", self.cup[cup_selected], self.spoon[1]]
            else: 
                self.decision = ["COFFEE BREWING","water", "coffee", self.milk[milk_selected],self.cup[cup_selected], self.spoon[1]]
        
        if sugar_decision == 1 and option_selected != 2: 
            sugar_selected = random.randint(0, 5) 
            self.decision.append(self.sugar[sugar_selected])
            
        print_decision(self.decision, False)
            
    def prepare_toast (self):

        option_selected = random.randint(0, 2)
        knife_selected = random.randint(0, 1)
        plate_selected = random.randint(0, 1)
        sliced_selected = random.randint(0, 1)
        
        if option_selected == 0: 
            self.decision = ["PREPARE SAVOURY TOAST", self.sliced_bread[sliced_selected], "olive oil","tomato sauce", self.plate[plate_selected], self.knife[knife_selected]]
        elif option_selected == 1: 
            jam_selected = random.randint(0, 2)
            self.decision = ["PREPARE SWEET TOAST", self.sliced_bread[sliced_selected], "butter", self.jam[jam_selected], self.plate[plate_selected], self.knife[knife_selected]]
        else: 
            self.decision = ["PREPARE SWEET TOAST", self.sliced_bread[sliced_selected], "nutella", self.plate[plate_selected], self.knife[knife_selected]]
        
        print_decision(self.decision, False)
    
    def prepare_cereals (self):

        milk_selected = random.randint(0, 1)
        spoon_selected = random.randint(0, 1)
        cereals_selected = random.randint(0, 2)
        bowl_selected = random.randint(0, 1)
        
        nesquick_decision = random.randint(0, 1)
        microwave_decision = random.randint(0, 1)
        
        self.decision = ["PREPARE CEREAL BOWL", self.bowl[bowl_selected], self.milk[milk_selected], self.cereals[cereals_selected], self.spoon[spoon_selected]]
        
        if nesquick_decision == 1: 
            self.decision.append("nesquick")
        if microwave_decision == 1: 
            self.decision.append("microwave")
        
        print_decision(self.decision, False)
        
def print_decision(decision, flag_write): 
    
    if flag_write==False:
        aster = "*" * len(decision[0])
        print("\n********"+aster+"************")
        print("***** ACTION: {} *****".format(decision[0]))
        print("********"+aster+"************\n")
        print("OBJECTS SELECTED:")
        for idx,obj in enumerate(decision[1:]):
            print(" - {}".format(obj))
        print("")
        save_history(decision)
    else:

           if os.path.isfile("breakfast_history_log.txt"):
               file = open("breakfast_history_log.txt", "a")
           else:
               file = open("breakfast_history_log.txt", "w") 

           timestr = time.strftime("%d/%m/%Y - %H:%M:%S")

           registro = "Date: "+timestr+"\n\n"
           file.write(registro)
           file.writelines('\n - '.join(decision[0:]))
           file.write("\n------------------------------------------------\n")
           file.close()


def save_history(decision): 

    print("Do you want to save this configuration? (y or n)")
    control = 1
    try:
        while control == 1: 
            data = input()
            if (data):
                if data=="y" or data=="Y": 
                    print_decision(decision, True)
                    control = 0
                elif data=="n" or data=="N":
                    print("The session was not save") 
                    control = 0
                else:
                    print("\nPlease introduced a valid answerd (y or n)\n")

    except:
      print("The session was not save") 
      
def action_decision (action): 
    objects = Objects()
    if (action == "drink"):
        objects.prepare_drink()
        
    elif (action == "toast"):
        objects.prepare_toast()
    elif (action == "cereals"): 
        objects.prepare_cereals()
    else:
        print("Please enter one of the following actions: \n drink, toast or cereals")
    
try:
  if (sys.argv[1]):
      action_decision(sys.argv[1])
except:
  print("Please enter one of the following actions (on the command line when the program is executed):\n drink, toast or cereals") 
  

