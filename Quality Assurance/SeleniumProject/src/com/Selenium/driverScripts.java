package com.Selenium;

import java.io.FileInputStream;

import org.apache.poi.xssf.usermodel.*;
import org.openqa.selenium.By;
//import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
//import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
//import org.openqa.selenium.chrome.ChromeOptions;
//import org.openqa.selenium.firefox.FirefoxDriver;

public class driverScripts {

	 public static void main(String[] args) {
		 
		/*Fire Box Driver
		 * 
		 * System.setProperty("webdriver.firefox.marionette",
		 * "D:\\Softwares\\Selenium\\D:\\Softwares\\Selenium\\geckodriver-v0.26.0-win64\\geckodriver.exe");
		 * WebDriver driver = new FirefoxDriver();
		 */
		
		String driverPath = "webdriver.chrome.driver";
	    String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe"; 
	    
	   	System.setProperty(driverPath,driverURL);
		/*
		 * For Headless Browser Testing 
		 * ChromeOptions options =new ChromeOptions();
		 *  options.setHeadless(true);
		 */
	   	
	    WebDriver driver = new ChromeDriver();	
	    String baseUrl = "https://www.facebook.com/";
	    driver.get(baseUrl);
	    try
	    {

	    /*************Get Commands *************/
	    
		/*
		 * System.out.println("Title : " + driver.getTitle());
		 * System.out.println("URL : " + driver.getCurrentUrl());
		 * //System.out.println("Page : " + driver.getPageSource()); String text
		 * =driver.findElement(By.xpath(".//*[@id=\"email\"]")).getText();
		 * System.out.println("Text : " + text);
		 */
		
		/***************** Browser Commands*****************/
		
		/*
		 * System.out.println("Current Browser : " + driver.getTitle());
		 * driver.navigate().to("https://www.amazon.in/");
		 * System.out.println("Current Browser : " + driver.getTitle());
		 * driver.navigate().back(); System.out.println("Current Browser : " +
		 * driver.getTitle()); driver.navigate().forward();
		 * System.out.println("Current Browser : " + driver.getTitle());
		 */
		
		 /***************** Validation Commands*****************/
		 
		/*
		 * WebElement email = driver.findElement(By.xpath(".//*[@id=\"email\"]"));
		 * if(email.isDisplayed() && email.isEnabled()) { email.sendKeys("Ashish");
		 * 
		 * }
		 * 
		 * WebElement pass = driver.findElement(By.xpath(".//*[@id=\"pass\"]")); if
		 * (pass.isDisplayed() && pass.isEnabled()) { pass.sendKeys("Tiwari"); }
		 * WebElement genderFemale =
		 * driver.findElement(By.xpath(".//*[@id=\"u_0_6\"]"));
		 * 
		 * if(genderFemale.isDisplayed() && (genderFemale.isSelected()==false)) {
		 * genderFemale.click();
		 * 
		 * }
		 */ 
		 
	    /******************** Locators ***********************/
	    
		/* 
		 * System.out.println( driver.findElement(By.id("email")));
		 * System.out.println( driver.findElement(By.name("email")));
		 * System.out.println( driver.findElement(By.cssSelector("input#u_0_m"))); //For Tag with ID // System.out.println(
		 * driver.findElement(By.cssSelector("input.inputtext login_form_input_box"))); //For Tag with class
		 * System.out.println(driver.findElement(By.cssSelector("input[name=email]"))); //For tag with attribute
		 * System.out.println(driver.findElement(By.xpath(".//*[@id=\"u_0_r\"]"))); }  
		 * 
		 * */	
	    	
	     System.out.println("***************** Try Block End **************************************************");
		 }
		 catch(Exception e)
		 {
			 System.out.println(e); 
		 }
		  finally
		  {
		  System.out.println("*************************************************************************");
		  System.out.println("Quiting Broswer");
		  driver.quit();
		  System.out.println("*************************************************************************");
		  }
	    }
	
}
