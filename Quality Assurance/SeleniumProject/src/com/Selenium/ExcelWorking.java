package com.Selenium;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.apache.poi.xssf.usermodel.*;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
//import org.openqa.selenium.support.ui.Select;

public class ExcelWorking {

	public static void main(String[] args) throws IOException,FileNotFoundException {
		
		String driverPath = "webdriver.chrome.driver";
	    String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe"; 
	    
	   	System.setProperty(driverPath,driverURL);
		
	    WebDriver driver  = new ChromeDriver();
	    String baseUrl = "https://sipcalculator.in/";
	    String sheetPath = "C:\\Users\\Ashish\\eclipse-workspace\\SIPData.xlsx";  
	    driver.get(baseUrl);
	    
	    FileInputStream file = new FileInputStream(sheetPath);
	    
	    XSSFWorkbook wordbook = new XSSFWorkbook(file);
	    XSSFSheet sheet = wordbook.getSheet("Sheet1");
	    
	    try {
	    int totalRow = sheet.getLastRowNum();
	    //int totalColumn = sheet.getRow(totalRow-1).getLastCellNum();
	    
	    for (int iterator=1;iterator<=totalRow;iterator++)
	    {
	    	XSSFRow currentRow  =sheet.getRow(iterator);
	    	
	    	XSSFCell monthInvst = currentRow.getCell(0);
	    	int monthlyInvestment =  (int) monthInvst.getNumericCellValue();
	    	
	    	int period = (int) currentRow.getCell(1).getNumericCellValue();
	    	int expected = (int) currentRow.getCell(2).getNumericCellValue();
	    	//String Adjust = (String) currentRow.getCell(3).getStringCellValue();
	    	
	    	
	    	WebElement ammount  = driver.findElement(By.id("amountInputField"));
	    	if(ammount.isDisplayed() && ammount.isEnabled()){
	    		ammount.sendKeys(String.valueOf(monthlyInvestment));}
	    	else{
	    		System.out.println("Monthly Investment Amount (Rs.) Not Found");}
	    	
	    	WebElement duration = driver.findElement(By.id("periodInputField"));
	    	if(duration.isDisplayed() && duration.isEnabled()) {
	    		duration.sendKeys(String.valueOf(period));
	    	}
	    	else{
	    		System.out.println("Investment Period Not Found");}
	    	
	    	WebElement annual = driver.findElement(By.id("returnRateInputField")); 
	    	if(annual.isDisplayed() && annual.isEnabled()){
	    		annual.sendKeys(String.valueOf(expected));}
	    	else{
	    		System.out.println("Expected Annual Returns (%) Not Found");}
	    	
	    	//Select inflation = new Select(driver.findElement(By.id("inflationSelectField")));
	    	//inflation.selectByValue(Adjust); 
	    	
	    	WebElement calculateButton = driver.findElement(By.id("calculateButton")); 
	    	if(calculateButton.isDisplayed() && calculateButton.isEnabled()){
	    		calculateButton.click();}
	    	else{
	    		System.out.println("Calculate button Not Found");}
	    	
	      	driver.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS);
	      	
	      	String finalValue  = driver.findElement(By.id("futureValue")).getText();
	      	System.out.println("Final Price  :"+finalValue);
	      	driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
	    	driver.navigate().back();	    	
	    }
		}
		catch(Exception e)
		{
			System.out.println("Expection Description : "+e);
		}
		finally {
			driver.quit();
			wordbook.close();
		}
	    
	}
}
