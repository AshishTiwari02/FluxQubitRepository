package com.Selenium;

import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

import com.android.dx.util.FileUtils;

import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

public class scrollUsingJavaScript {

	public static void main(String[] args) {
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";

		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver(); 
		String baseUrl = "https://en.wikipedia.org/wiki/Data";
		try
		{	
			System.out.println("************* Try Block ***********************");
			 driver.get(baseUrl);
			 driver.manage().window().maximize();
			 JavascriptExecutor js  = (JavascriptExecutor) driver;
			 
			 //1. Scroll By window
			// js.executeScript("window.scrollBy(0,1000)" ,"");
			 Thread.sleep(3000);
			 
			 //2. Scroll till found some data
			 WebElement flag = driver.findElement(By.xpath(".//*[@id=\"mw-content-text\"]/div/div[6]/div/a/img"));
			 js.executeScript("arguments[0].scrollIntoView();", flag);
			 Thread.sleep(3000);
			 
			 //3. scroll till page end
			 js.executeScript("window.scrollTo(0,document.body.scrollHeight)");
			 Thread.sleep(3000);
			 
			 
			 //Take screenshot
			 
			/*
			 * java.io.File imgSrc = ((TakesScreenshot)
			 * driver).getScreenshotAs(OutputType.FILE); File trg = new File("");
			 * FileUtils.copyFile(imgSrc,trg);
			 */ 
			 
			 
			 Connection con = DriverManager.getConnection("","","");
			 Statement stmt  = con.createStatement();
			 String s = "Select * From DAta";
			 stmt.executeQuery(s);
			 con.close();
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
			 
		}
		catch(Exception e)
		{
			System.out.println("************* Catch Block ***********************");
			System.out.println("Exception Found in Scripts " + e);
		}
		finally
		{
		   System.out.println("************* Finally Block ***********************");	
		   driver.close();
		   driver.quit();
		 }
		

	}

}
