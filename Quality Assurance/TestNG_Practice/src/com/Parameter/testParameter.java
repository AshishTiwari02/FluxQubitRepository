package com.Parameter;

import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.annotations.Test;

public class testParameter {
	WebDriver driver;
	
	@Test
	public void setUp() throws InterruptedException
	{		
			String driverType = "webdriver.chrome.driver";
			String driverUrl = "D:\\Softwares\\Selenium\\chromedriver\\chromedriver.exe";
			System.setProperty(driverType,driverUrl);
			driver = new ChromeDriver();
			String urlPath ="https://www.google.co.in/";
			driver.get(urlPath);
			driver.manage().window().maximize();
			driver.manage().deleteAllCookies();
			driver.manage().timeouts().pageLoadTimeout(2000, TimeUnit.MILLISECONDS);
			
			WebElement b = driver.findElement(By.xpath(".//*[@id=\"tsf\"]/div[2]/div[1]/div[1]/div/div[2]/input"));
			b.sendKeys("");
	}
	
	
	
	
	
	
}
