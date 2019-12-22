package com.Selenium;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class demoWebButton {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String driverPath = "webdriver.chrome.driver";
	    String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe"; 
	    
	    System.setProperty(driverPath, driverURL);
	    
	    WebDriver driver  = new ChromeDriver();
	    String baseUrl = "D:\\Practice\\HTML\\Sample.html"; 
	    driver.get(baseUrl);
		
	    int totalRow = driver.findElements(By.xpath("html/body/table/tbody/tr/td/table/tbody/tr")).size();
	    int totalColumn = driver.findElements(By.xpath("html/body/table/tbody/tr/td/table/tbody/tr/th")).size();
	    System.out.println("Total Row : "+totalRow);
	    System.out.println("Total Column : "+totalColumn);
	    
	    for (int i=2;i<=totalRow;i++)
	    {
	    	for(int j=1;j<=totalColumn;j++)
	    	{
	    		System.out.print(driver.findElement(By.xpath("html/body/table/tbody/tr/td/table/tbody/tr["+i+"]/th["+j+"]")).getText());
	    	}
	    	System.out.println();
	    }
	    
	    
	    
	    driver.quit();
	}

}
