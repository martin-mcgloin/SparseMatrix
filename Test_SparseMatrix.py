#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:04:28 2018

@author: martinmcgloin
"""

from  scipy import *
from  pylab import *
from SparseMatrix import *
import unittest

class TestSparseMatrix(unittest.TestCase):
     
    def setUp(self):
        self.matrix1 = numpy.array([
            [5,4,3,2,1,0],
            [0,5,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ])
            
        self.matrix2 = numpy.array([
            [10,0,0,0,0,0],
            [0,-5,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,9]
        ])
        
        self.matrix3 = numpy.array([
            [1,0,0,0,5,0,0],
            [0,0,3,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,2,0,0,0,6,0],
            [0,0,0,4,0,0,0]
        ])
    
        self.matrix4 = numpy.array([
            [-1,2,0,0,-5,0,0],
            [0,0,-3,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]
        ])
        
        self.matrix5 = numpy.array([
                [3,4,5],
                [0,3,0],
                [6,6,8]
        ])
    
        self.matrix6 = numpy.array([
                [1,-1,2],
                [0,-3,1]
        ])
    
        self.vector1 = numpy.array([1,3,0]) #[15,9,24]
        self.vector2 = numpy.array([2,1,0]) # [1,-3]
        self.matrix_empty = numpy.zeros(shape=(100,100))
        self.matrix_100 = numpy.ones(shape=(10,10))
        self.zero_vector = numpy.zeros(shape=(7,1))
        
        self.sparseMatrix1=SparseMatrix(self.matrix1)
        self.sparseMatrix2=SparseMatrix(self.matrix2)
        self.sparseMatrix3=SparseMatrix(self.matrix3)
        self.sparseMatrix4=SparseMatrix(self.matrix4)
        self.sparseMatrix5=SparseMatrix(self.matrix5)
        self.sparseMatrix6=SparseMatrix(self.matrix6)
        self.sparseMatrixZero=SparseMatrix(self.matrix_empty)
        self.sparseMatrixOnes=SparseMatrix(self.matrix_100)
        
        pass
    
    def tearDown(self):
        pass
        
        
    def test_count_non_zero(self):
        self.assertEqual(0,self.sparseMatrixZero.number_of_nonzero)
        self.assertEqual(0,len(self.sparseMatrixZero.values))
        self.assertEqual(100,SparseMatrix(self.matrix_100).number_of_nonzero)
        self.assertEqual(100,len(SparseMatrix(self.matrix_100).values))
        
    def test_tolerance(self):
        self.assertEqual(0,SparseMatrix(self.matrix_100,3).number_of_nonzero)
        self.assertEqual(100,SparseMatrix(self.matrix_100,0.5).number_of_nonzero)
        self.assertEqual(4,SparseMatrix(self.matrix1,2).number_of_nonzero)
    
    def test_change_value(self):
        self.assertEqual(0,self.sparseMatrixZero.number_of_nonzero)
        self.sparseMatrixZero.change_value(1,1,5)
        self.assertEqual(1,self.sparseMatrixZero.number_of_nonzero)
        self.assertEqual(5,self.sparseMatrixZero.values[0])
        
    def test_mul_input_validation(self):
        with self.assertRaises(TypeError):
            self.sparseMatrix6*"a string"
        with self.assertRaises(TypeError):
            self.sparseMatrix6*12345
        
    def test_mul(self):
        self.assertTrue(np.array_equal([1,-3],
                                       self.sparseMatrix6*self.vector2))
        self.assertTrue(np.array_equal([15,9,24],
                                       self.sparseMatrix5*self.vector1))
    def test_mul_zero_vector(self):
        self.assertTrue(np.array_equal(self.sparseMatrix4*self.zero_vector,
                                       [0]*5))    
        
    def test_add_input_validation(self):
        with self.assertRaises(TypeError):
            self.sparseMatrix6*"a string"
        with self.assertRaises(TypeError):
            self.sparseMatrix6*12345 
        with self.assertRaises(ValueError):
            self.sparseMatrix1+self.sparseMatrix6

    def test_add(self):
        self.sparseMatrix1+self.sparseMatrix2
        self.assertEqual([15,4,3,2,1,9],self.sparseMatrix1.values)
        self.sparseMatrix3+self.sparseMatrix4
        self.assertEqual(self.sparseMatrix3.values,[2,2,6,4])
        
    def test_convert_to_CSC(self):
        self.sparseMatrix5.convert_to_CSC()
        self.assertEqual(self.sparseMatrix5.CSC_values,[3,6,4,3,6,5,8])
        self.assertEqual(self.sparseMatrix5.CSC_row_indicies,[0,2,0,1,2,0,2])
        self.assertEqual(self.sparseMatrix5.CSC_col_extent,[0,2,5,7])
        self.sparseMatrix3.convert_to_CSC()
        self.assertEqual(self.sparseMatrix3.CSC_values,[1,2,3,4,5,6])
        self.assertEqual(self.sparseMatrix3.CSC_row_indicies,[0,3,1,4,0,3])
        self.assertEqual(self.sparseMatrix3.CSC_col_extent,[0,1,2,3,4,5,6])
        
        

unittest.main(argv=['first-arg-is-ignored'], exit=False)
