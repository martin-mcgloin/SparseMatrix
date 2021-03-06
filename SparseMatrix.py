#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import bisect
  
class SparseMatrix:
    CSR = "CSR"
    CSC = "CSC"
    #CSR Values
    values = []
    col_indicies = []
    row_extent = []
    #CSC Values
    CSC_values = []
    CSC_row_indicies = []
    CSC_col_extent= []
    Tol = 0
    
    def __init__(self, matrix, tolerance=0):
        self.__check_valid_arguments_init(matrix)       
        self.intern_represent = self.CSR
        self.Tol=tolerance
        self.__generate_CSR(matrix, tolerance)
        self.__update_non_zero_count()
    
    def __generate_CSR(self, matrix, tolerance):
        """Converts a numpy.darray matrix into a CSR formatted 
        
        Iterates through all values in matrix left to right and stores only 
        non-zero values larger than tolerance (defaults to 0) in CSR vectors. 

        Args:
            matrix (numpy.darray): the sparse matrix to use to instantiate the 
            SparseMatrix class and store in CSR format.
            
        Returns:
            nothing         
        """        
        values = []
        row_extent = [0]
        col_indicies = []
        matrix_rows = matrix.shape[0]
        matrix_cols = matrix.shape[1]
        for i in range(0, matrix_rows):
            for j in range(0,matrix_cols):
                if (abs(matrix[i][j]) > tolerance):
                    values.append(matrix[i][j])
                    col_indicies.append(j)            
            row_extent.append(len(values))
        
        self.values=values
        self.col_indicies=col_indicies
        self.row_extent=row_extent
     
        
    def change_value(self, row, col, value):
        """Changes the value at the given row and column index
        
        Changes the value in the CSR representation of the sparse matrix if 
        value already exists, removes the value if the new value is 0, or inserts
        a new value in the CSR if value was previously zero.
        
        Args:
            row (int): the index row position of the new value
            
            col (int): the index col position of the new value
            
            value (float): the new value
            
        Returns:
            nothing
        """       
        row_col_indicies = self.__get_row_col_indicies(row)  
        #If value exist at location, replace, else, insert new value
        if col in row_col_indicies:
            index = row_col_indicies.index(col)+self.row_extent[row]          
            if value !=0:
                self.values[index]=value
            #If new value is zero - remove value          
            else:           
                self.values.pop(index)
                self.col_indicies.pop(index)
                self.__update_non_zero_count()
                for j in range(row+1, len(self.row_extent)):
                    self.row_extent[j]-=1                
        else:
            insert_index = self.row_extent[row]+len(row_col_indicies)
            for col_id in row_col_indicies:
                if col_id > col:
                    insert_index = self.row_extent[row]+row_col_indicies.index(col_id)
                    break
            
            self.values.insert(insert_index,value)
            self.col_indicies.insert(insert_index,col)
            for j in range(row+1, len(self.row_extent)):
                self.row_extent[j]+=1
            self.__update_non_zero_count()    
                
    
    def convert_to_CSC(self):   
        """Converts sparse matrix in CSR to CSC format
        
        Iterates through all values in CSR and retrievs original matrix row
        and col index for value, then places in a CSC format based vectors.
        
        Args:
            nothing
            
        Returns:
            nothing
        """
        if self.intern_represent != self.CSR:
            return
        CSC_values =[]
        CSC_row_indicies=[]
        CSC_col_extent=[0]* (max(self.col_indicies)+2)
            
        for index in range(0,len(self.values)):
            value = self.values[index]
            col = self.col_indicies[index]
            row = self.__get_row_index(index)
           
            start = CSC_col_extent[col]
            stop = CSC_col_extent[col+1]
            insert_index = bisect.bisect(CSC_row_indicies,row, start, stop)
            CSC_values.insert(insert_index, value)
            CSC_row_indicies.insert(insert_index, row)
            for j in range(col+1, len(CSC_col_extent)):
                CSC_col_extent[j]+=1
        self.CSC_values = CSC_values
        self.CSC_row_indicies = CSC_row_indicies
        self.CSC_col_extent= CSC_col_extent
        
    def compare_matricies(self, other):
        """Compares two SparseMatrix instances for equality
        
        Compares that two SparseMatrix classes in CSR format have the exact 
        same values. Initially compares row size and number of values, then 
        iterates and compares each value and column position.
        
        Args:
            other(SparseMatrix): the SparseMatrix instance to compare with.
            
        Returns:
            boolean: True of false depending on result of comparison.
        """
        self.__check_valid_argument_compare_matricies(other)
        
        if ((len(self.row_extent) != len(other.row_extent)) or
            (len(self.values) != len(other.values))):
            return False
        for i in range(0,len(self.values)):
            if (self.values[i] != other.values[i] and 
                self.col_indicies[i] != other.col_indicies[i]):
                return False
        return True
    
    def __mul__(self,other):   
        """Implements Matrix Multiplication with 1-dim vector
        
        Multiplies SparseMatrix with a numpy.array of rank 1 (vector) and 
        same lenght as number of columns in SparseMatrix instance.
        
        Args:
            other (array): The one-dimensional vector to multiply SparseMatrix
            with.
            
        Returns:
            array: The return value, an array of size equal to number of rows 
            in Sparse Matrix     
        """
        self.__check_valid_arguments_mul(other)
        result = numpy.zeros(len(self.row_extent)-1)
        for index in range(0, len(self.values)):
            value = self.values[index]
            col_index = self.col_indicies[index]
            row_index = self.__get_row_index(index)
            multiplier = other[col_index]
            result[row_index] += value*multiplier
        return result
        
    
    def __add__(self,other):
        """Implements Matrix Addition 
        
        Elementwise addition of values in SparseMatrix with elements with same 
        row and col index as in 'other' instance. Values in SparseMatrix 
        instance (self) are updated.
        
        If result of addition equals 0, element is removed from SparseMatrix
        instance and number_of_nonzero is updated. 
        
        Args:
            other (SparseMatrix): SparseMatrix of same shape as SparseMatrix 
            instance to add to SparseMatrix.
            
        Returns:
            nothing     
        """
        self.__check_valid_argument_add(other)        
        for index in range(0,len(other.values)):
            addition_value = other.values[index]
            col_index = other.col_indicies[index]
            row = other.__get_row_index(index)
            current_value = self.__get_value(row,col_index)
            if current_value == 0:
                self.change_value(row,col_index,addition_value)
            else:
                self.change_value(row,col_index, current_value+addition_value)
    
    #Private helper functions for the class
    
    def __str__(self):
        return "{0}\n{1}\n{2}".format(self.values,self.row_extent,self.col_indicies)
                            
    def __update_non_zero_count(self):
        self.number_of_nonzero = self.row_extent[-1]
        
    def __get_value(self,row,col):
        col_values = self.__get_row_col_indicies(row)
        if col not in col_values:
            return 0
        else:
            value_index = col_values.index(col)
            return self.__get_row_values(row)[value_index]

    def __get_row_values(self,row):
        start_index = self.row_extent[row]
        stop_index = self.row_extent[row+1]
        return self.values[start_index:stop_index]
    
    def __get_row_col_indicies(self,row):
        start_index = self.row_extent[row]
        stop_index = self.row_extent[row+1]
        return self.col_indicies[start_index:stop_index]
    
    def __get_row_index(self,value_index):
        return bisect.bisect(self.row_extent,value_index)-1
    
    def __check_valid_argument_add(self,other):
        if not isinstance(other, SparseMatrix):
            raise TypeError("unsupported type for addition with SparseMatrix:",
                            type(other), 
                            ". Can only add SparseMatrix with SparseMatrix.")
        if len(self.row_extent) != len(other.row_extent):
            raise ValueError("Can only add SparseMatricies of same shape") 
              
    def __check_valid_arguments_mul(self,other):
        if not isinstance(other, numpy.ndarray):
            raise TypeError("unsupported type for multiplication. ",
                            "Can only multiply SparseMatrix with array.")
        
    def __check_valid_arguments_init(self,matrix):
        if not isinstance(matrix, numpy.ndarray):
            raise ValueError("You can only generate Sparse Matrix from a numpy array")
            
    def __check_valid_argument_compare_matricies(self,other):
        if not isinstance(other, SparseMatrix):
            raise TypeError("unsupported type for comparison with SparseMatrix:",
                            type(other), 
                            ". Can only compare SparseMatrix.")
        
