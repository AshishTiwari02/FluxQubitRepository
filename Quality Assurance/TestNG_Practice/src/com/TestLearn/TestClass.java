package com.TestLearn;

//import org.openqa.selenium.By;
//import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class TestClass {

	static WebDriver driver;
	
	public static void main(String[] args) {
		try {
		String driverType = "webdriver.chrome.driver";
		String driverUrl = "D:\\Softwares\\Selenium\\chromedriver\\chromedriver.exe";
		System.setProperty(driverType,driverUrl);
	
		driver = new ChromeDriver();
	    String url ="https://mail.google.com/mail/u/0/#inbox";
	    driver.get(url);
	    
	    
		}
		catch(Exception e) {
			//System.out.println("Exception found : "+e);
		}
	   
	    finally {
	    	
	    	driver.close();
	 	    driver.quit();
	    }
	    
}
}
