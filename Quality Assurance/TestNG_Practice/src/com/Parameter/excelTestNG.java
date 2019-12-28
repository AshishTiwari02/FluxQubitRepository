package com.Parameter;

import java.util.concurrent.TimeUnit;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class excelTestNG {

WebDriver driver;	

@BeforeMethod
public void setup() throws InterruptedException
{
	String driverType = "webdriver.chrome.driver";
	String driverUrl = "D:\\Softwares\\Selenium\\chromedriver\\chromedriver.exe";
	System.setProperty(driverType,driverUrl);
	driver = new ChromeDriver(); 
	driver.manage().window().maximize();
	driver.manage().timeouts().pageLoadTimeout(40, TimeUnit.SECONDS);
	driver.manage().timeouts().implicitlyWait(40, TimeUnit.SECONDS);
	Thread.sleep(2000);
	
	driver.get("https://www.facebook.com/");
}
	
	@Test
	public void checkFacebook()
	{
		
		
		
	}
	
	
}
