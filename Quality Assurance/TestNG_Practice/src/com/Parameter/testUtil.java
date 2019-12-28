package com.Parameter;

import java.io.FileInputStream;
import java.io.FilterInputStream;
import java.util.ArrayList;

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;


public class testUtil {
	
	public static ArrayList<Object[]> getDataExcel()
	{
		
		ArrayList<Object[]> mydata = new ArrayList<Object[]>();
		try {
			 FileInputStream file = new FileInputStream("C:\\Users\\Ashish\\eclipse-workspace\\UserFile.xlsx");
			    
			    XSSFWorkbook wordbook = new XSSFWorkbook(file);
			    XSSFSheet currentSheet = wordbook.getSheet("Sheet1");
			    int rowNum= currentSheet.getLastRowNum();
			    
			    for (int i=1;i<rowNum;i++)
			    {

			    	XSSFRow currentRow  =currentSheet.getRow(i);
			    	//XSSFCell currentCell = currentRow.getCell(0);
			    	XSSFCell username = currentRow.getCell(0);
			    	
			    	
			    }
			    			
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		
		
		
		
		return mydata;
	}
	
	
	
}
